import cv2
import numpy as np
from scipy import stats


def order_calibration_points(points):
    """Return points ordered as top-left, top-right, bottom-right, bottom-left."""
    if len(points) != 4:
        return []

    points = [(float(x), float(y)) for x, y in points]
    sorted_pts = sorted(points, key=lambda p: p[1])
    top = sorted(sorted_pts[:2], key=lambda p: p[0])
    bottom = sorted(sorted_pts[2:], key=lambda p: p[0])
    return [top[0], top[1], bottom[1], bottom[0]]


def get_calibration_bounds(calibration_points, width, height):
    """Clamp calibration bounds to valid image coordinates."""
    if len(calibration_points) != 4 or width <= 0 or height <= 0:
        return None

    xs = [p[0] for p in calibration_points]
    ys = [p[1] for p in calibration_points]

    min_x = max(0, min(int(np.floor(x)) for x in xs))
    max_x = min(width, max(int(np.ceil(x)) for x in xs))
    min_y = max(0, min(int(np.floor(y)) for y in ys))
    max_y = min(height, max(int(np.ceil(y)) for y in ys))

    if min_x >= max_x or min_y >= max_y:
        return None

    return min_x, max_x, min_y, max_y


def apply_exclusion_zones(mask, exclusion_zones):
    """Apply rectangle exclusion zones to a binary mask in-place."""
    if exclusion_zones is None:
        return mask

    h, w = mask.shape[:2]
    for x0, y0, x1, y1 in exclusion_zones:
        ex0 = max(0, min(w - 1, int(np.floor(min(x0, x1)))))
        ey0 = max(0, min(h - 1, int(np.floor(min(y0, y1)))))
        ex1 = max(0, min(w - 1, int(np.ceil(max(x0, x1)))))
        ey1 = max(0, min(h - 1, int(np.ceil(max(y0, y1)))))
        if ex1 >= ex0 and ey1 >= ey0:
            cv2.rectangle(mask, (ex0, ey0), (ex1, ey1), 0, -1)
    return mask


def auto_calibrate_image(image_array):
    """Detect graph area corners from image contours."""
    try:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2,
        )
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect).astype(int)
        return order_calibration_points(box)
    except Exception:
        return None


def enhance_line_detection(image_array, target_color, tolerance):
    """Create a binary mask isolating target color within Euclidean tolerance."""
    img_float = image_array.astype(np.float32)
    target_color_float = np.array(target_color, dtype=np.float32)
    dist_sq = np.sum((img_float - target_color_float) ** 2, axis=-1)
    mask = (dist_sq <= float(tolerance) ** 2).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def find_line_y_in_column(column_data, last_y=None, max_thickness=15):
    """Find the most probable line center in one mask column."""
    y_coords_detected = np.where(column_data > 0)[0]
    if y_coords_detected.size == 0:
        return None

    diffs = np.diff(y_coords_detected)
    segments = np.split(y_coords_detected, np.where(diffs > 1.5)[0] + 1)

    valid_segments = [s for s in segments if s.size > 0 and s.size <= max_thickness]
    if not valid_segments:
        return None

    centers = [int(np.mean(s)) for s in valid_segments]
    if last_y is None:
        return centers[0]

    closest_y = min(centers, key=lambda c: abs(c - last_y))
    if abs(closest_y - last_y) > 50:
        return None
    return closest_y


def extract_points_method1(image_array, mask, calibration_points, exclusion_zones=None):
    """Extract one Y point per X column using weighted grayscale centering."""
    if image_array is None or mask is None:
        return []

    h, w = image_array.shape[:2]
    bounds = get_calibration_bounds(calibration_points, w, h)
    if not bounds:
        return []
    min_x, max_x, min_y, max_y = bounds

    working_mask = mask.copy()
    apply_exclusion_zones(working_mask, exclusion_zones)

    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    points = []

    for x in range(min_x, max_x):
        col = working_mask[min_y:max_y, x]
        y_idx = np.where(col > 0)[0]
        if y_idx.size == 0:
            continue

        y_coords = y_idx + min_y
        weights = 255.0 - gray[y_coords, x].astype(np.float32)
        if np.sum(weights) > 0:
            y_center = float(np.sum(y_coords * weights) / np.sum(weights))
        else:
            y_center = float(np.mean(y_coords))
        points.append((float(x), y_center))

    return points


def build_method2_mask(image_array, target_color, tolerance, calibration_points):
    """Create masked plot region for method 2 sampling."""
    if image_array is None:
        return None

    bgr_color = np.array([target_color[2], target_color[1], target_color[0]], dtype=np.int16)
    tol = int(tolerance)
    lower = np.clip(bgr_color - tol, 0, 255).astype(np.uint8)
    upper = np.clip(bgr_color + tol, 0, 255).astype(np.uint8)

    bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    color_mask = cv2.inRange(bgr_image, lower, upper)

    plot_mask = np.zeros_like(color_mask)
    cv2.fillPoly(plot_mask, [np.array(calibration_points, dtype=np.int32)], 255)

    kernel = np.ones((3, 3), np.uint8)
    eroded_plot_mask = cv2.erode(plot_mask, kernel, iterations=2)
    return cv2.bitwise_and(color_mask, eroded_plot_mask)


def extract_points_method2(
    final_mask,
    calibration_points,
    exclusion_zones=None,
    mode="step",
    value=10,
):
    """Extract points by column sampling with continuity-aware line tracking."""
    if final_mask is None:
        return []

    h, w = final_mask.shape[:2]
    bounds = get_calibration_bounds(calibration_points, w, h)
    if not bounds:
        return []
    min_x, max_x, min_y, max_y = bounds

    if mode == "count":
        num_points = max(2, int(value))
        x_values = np.linspace(min_x, max_x - 1, num_points, dtype=int)
    else:
        step = max(1, int(value))
        x_values = range(min_x, max_x, step)

    exclusion_zones = exclusion_zones or []
    pixel_points = []
    last_y = None

    for x in x_values:
        if x < 0 or x >= w:
            continue
        y = find_line_y_in_column(final_mask[:, x], last_y)
        if y is None:
            continue

        blocked = False
        for x0, y0, x1, y1 in exclusion_zones:
            if min(x0, x1) <= x <= max(x0, x1) and min(y0, y1) <= y <= max(y0, y1):
                blocked = True
                break
        if blocked:
            continue

        pixel_points.append((int(x), int(y)))
        last_y = y

    # Keep extremal points in the plot box if present.
    top_hit = None
    bottom_hit = None
    for y_scan in range(min_y, max_y):
        row = np.where(final_mask[y_scan, min_x:max_x] > 0)[0]
        if row.size > 0:
            top_hit = (int(min_x + row[0]), int(y_scan))
            break

    for y_scan in range(max_y - 1, min_y - 1, -1):
        row = np.where(final_mask[y_scan, min_x:max_x] > 0)[0]
        if row.size > 0:
            bottom_hit = (int(min_x + row[0]), int(y_scan))
            break

    for candidate in (top_hit, bottom_hit):
        if candidate is None:
            continue
        if not any(abs(p[0] - candidate[0]) < 5 and abs(p[1] - candidate[1]) < 5 for p in pixel_points):
            pixel_points.append(candidate)

    pixel_points.sort(key=lambda p: p[0])
    return pixel_points


def convert_to_real_coordinates(extracted_points, calibration_points, x_min, x_max, y_min, y_max):
    """Convert pixel points to graph coordinates with perspective transform."""
    if len(calibration_points) != 4 or not extracted_points:
        return []

    src = np.array(calibration_points, dtype="float32")
    dst = np.array([[x_min, y_max], [x_max, y_max], [x_max, y_min], [x_min, y_min]], dtype="float32")

    matrix = cv2.getPerspectiveTransform(src, dst)
    real_coords = cv2.perspectiveTransform(np.array([extracted_points], dtype="float32"), matrix)[0]
    return real_coords.tolist()


def calculate_stats(real_coordinates):
    """Calculate summary statistics for extracted graph coordinates."""
    if not real_coordinates:
        return {}

    x_vals = [p[0] for p in real_coordinates]
    y_vals = [p[1] for p in real_coordinates]

    stats_dict = {
        "Points": len(real_coordinates),
        "X Mean": np.mean(x_vals),
        "X Std Dev": np.std(x_vals),
        "Y Mean": np.mean(y_vals),
        "Y Std Dev": np.std(y_vals),
        "Correlation": np.corrcoef(x_vals, y_vals)[0, 1] if len(x_vals) > 1 else 0,
        "Area": np.trapz(y_vals, x_vals) if len(x_vals) > 1 else 0,
    }
    return stats_dict


def calculate_stats_extended(real_coordinates):
    """Calculate extended stats used by the desktop UI."""
    if not real_coordinates:
        return {}

    x_vals = [p[0] for p in real_coordinates]
    y_vals = [p[1] for p in real_coordinates]

    return {
        "Points": len(real_coordinates),
        "X Mean": np.mean(x_vals),
        "X Std Dev": np.std(x_vals),
        "X Skew": stats.skew(x_vals),
        "X Kurtosis": stats.kurtosis(x_vals),
        "Y Mean": np.mean(y_vals),
        "Y Std Dev": np.std(y_vals),
        "Y Skew": stats.skew(y_vals),
        "Y Kurtosis": stats.kurtosis(y_vals),
        "Correlation": np.corrcoef(x_vals, y_vals)[0, 1] if len(x_vals) > 1 else 0,
        "Area (Trapezoid)": np.trapz(y_vals, x_vals) if len(x_vals) > 1 else 0,
    }
