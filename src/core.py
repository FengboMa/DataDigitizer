import numpy as np
import cv2
from scipy import stats

def auto_calibrate_image(image_array):
    """
    Detects the graph area in the image using contour detection.
    Returns a list of 4 points (tuples) or None if detection fails.
    """
    try:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = box.astype(int)
        
        # Sort points: top-left, top-right, bottom-right, bottom-left
        points = sorted(box, key=lambda p: p[1])
        top_points = sorted(points[:2], key=lambda p: p[0])
        bottom_points = sorted(points[2:], key=lambda p: p[0])
        
        return [tuple(top_points[0]), tuple(top_points[1]), tuple(bottom_points[1]), tuple(bottom_points[0])]
    except Exception:
        return None

def enhance_line_detection(image_array, target_color, tolerance):
    """
    Creates a binary mask isolating the target color within tolerance.
    """
    img_float = image_array.astype(np.float32)
    target_color_float = np.array(target_color, dtype=np.float32)
    dist_sq = np.sum((img_float - target_color_float) ** 2, axis=-1)
    mask = (dist_sq <= tolerance ** 2).astype(np.uint8) * 255
    
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

def find_line_y_in_column(column_data, last_y=None, max_thickness=15):
    """
    Finds the center Y-coordinate of a line segment in a single column slice.
    """
    y_coords_detected = np.where(column_data > 0)[0]
    if y_coords_detected.size == 0:
        return None

    diffs = np.diff(y_coords_detected)
    segments = np.split(y_coords_detected, np.where(diffs > 1.5)[0] + 1)
    
    if not segments or segments[0].size == 0:
        return None
        
    valid_segments = []
    for s in segments:
        if s.size > 0 and s.size <= max_thickness:
            valid_segments.append(s)

    if not valid_segments:
        return None

    segment_centers = [int(np.mean(s)) for s in valid_segments]
    
    if last_y is None:
        return segment_centers[0]

    closest_y = -1
    min_dist = float('inf')
    for y_center in segment_centers:
        dist = abs(y_center - last_y)
        if dist < min_dist:
            min_dist = dist
            closest_y = y_center
    
    if min_dist > 50:
        return None 

    return closest_y

def convert_to_real_coordinates(extracted_points, calibration_points, x_min, x_max, y_min, y_max):
    """
    Converts pixel coordinates to real graph coordinates using perspective transform.
    """
    if len(calibration_points) != 4 or not extracted_points:
        return []
        
    src = np.array(calibration_points, dtype='float32')
    dst = np.array([[x_min, y_max], [x_max, y_max], [x_max, y_min], [x_min, y_min]], dtype='float32')
    
    matrix = cv2.getPerspectiveTransform(src, dst)
    real_coords = cv2.perspectiveTransform(np.array([extracted_points], dtype='float32'), matrix)[0]
    
    return real_coords.tolist()

def calculate_stats(real_coordinates):
    """
    Calculates statistical metrics for the extracted data.
    """
    if not real_coordinates:
        return {}
        
    x_vals = [p[0] for p in real_coordinates]
    y_vals = [p[1] for p in real_coordinates]
    
    stats_dict = {
        'Points': len(real_coordinates),
        'X Mean': np.mean(x_vals), 'X Std Dev': np.std(x_vals),
        'Y Mean': np.mean(y_vals), 'Y Std Dev': np.std(y_vals),
        'Correlation': np.corrcoef(x_vals, y_vals)[0, 1] if len(x_vals) > 1 else 0,
        'Area': np.trapz(y_vals, x_vals) if len(x_vals) > 1 else 0
    }
    return stats_dict
