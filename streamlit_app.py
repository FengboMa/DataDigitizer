import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.elements.image as st_image
from PIL import Image, ImageColor
from streamlit_drawable_canvas import st_canvas

from src import core

# Compatibility shim for streamlit-drawable-canvas with newer Streamlit.
if not hasattr(st_image, "image_to_url"):
    from streamlit.elements.lib.image_utils import image_to_url as _image_to_url_new
    from streamlit.elements.lib.layout_utils import LayoutConfig

    def _image_to_url_legacy(image, width, clamp, channels, output_format, image_id):
        return _image_to_url_new(
            image,
            LayoutConfig(width=width),
            clamp,
            channels,
            output_format,
            image_id,
        )

    st_image.image_to_url = _image_to_url_legacy

st.set_page_config(page_title="Raman Data Digitizer", layout="wide")
st.title("Raman Data Digitizer (Web)")

if "result" not in st.session_state:
    st.session_state.result = None
if "image_id" not in st.session_state:
    st.session_state.image_id = None
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0


def _score_points(points, x_bounds):
    if not points:
        return {"points": 0, "x_coverage_pct": 0.0, "score": 0}
    x_vals = [p[0] for p in points]
    span = max(1.0, x_bounds[1] - x_bounds[0])
    coverage = max(0.0, min(1.0, (max(x_vals) - min(x_vals)) / span))
    density = max(0.0, min(1.0, len(points) / 250.0))
    score = int(round((0.7 * coverage + 0.3 * density) * 100))
    return {
        "points": len(points),
        "x_coverage_pct": round(coverage * 100, 1),
        "score": score,
    }


def _prepare_replot_df(df, resample_count, smooth_window):
    working = df.sort_values("X").groupby("X", as_index=False)["Y"].mean()

    if resample_count > 1 and len(working) > 1:
        x_new = np.linspace(working["X"].min(), working["X"].max(), int(resample_count))
        y_new = np.interp(x_new, working["X"].to_numpy(), working["Y"].to_numpy())
        working = pd.DataFrame({"X": x_new, "Y": y_new})

    if smooth_window > 1:
        working["Y"] = working["Y"].rolling(window=int(smooth_window), center=True, min_periods=1).mean()

    return working


def _overlay_figure(image, m1_points=None, m2_points=None, active_points=None):
    fig = px.imshow(np.array(image))
    fig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=30, b=0), legend_title_text="Overlay")
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    if m1_points:
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in m1_points],
                y=[p[1] for p in m1_points],
                mode="lines",
                line=dict(color="#14b8a6", width=1.8),
                name="Method 1",
                opacity=0.9,
            )
        )
    if m2_points:
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in m2_points],
                y=[p[1] for p in m2_points],
                mode="lines",
                line=dict(color="#f97316", width=1.8),
                name="Method 2",
                opacity=0.9,
            )
        )
    if active_points:
        fig.add_trace(
            go.Scatter(
                x=[p[0] for p in active_points],
                y=[p[1] for p in active_points],
                mode="markers",
                marker=dict(size=3, color="#ef4444"),
                name="Active points",
            )
        )
    return fig


with st.sidebar:
    st.header("1. Configuration")
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "bmp"])

    st.header("2. Axes Calibration")
    col1, col2 = st.columns(2)
    x_min = col1.number_input("X Min", value=0.0)
    x_max = col2.number_input("X Max", value=100.0)
    y_min = col1.number_input("Y Min", value=0.0)
    y_max = col2.number_input("Y Max", value=100.0)

    st.header("3. Extraction Settings")
    line_color_hex = st.color_picker("Line Color", "#000000")
    color_tolerance = st.slider("Color Tolerance", 0, 150, 50)
    extraction_mode = st.selectbox("Primary Method", ["Method 1 (Color Mask)", "Method 2 (Advanced)"])
    run_both_methods = st.checkbox("Run both methods for comparison", value=True)

    sample_mode = st.selectbox("Method 2 Sampling", ["Step", "Number of points"])
    sample_value = st.number_input(
        "Method 2 Value",
        min_value=1,
        value=10,
        help="Step size in pixels or number of points, depending on sampling mode.",
    )

if not uploaded_file:
    st.info("Please upload an image to start.")
    st.stop()

image_id = f"{uploaded_file.name}:{uploaded_file.size}"
if st.session_state.image_id != image_id:
    st.session_state.image_id = image_id
    st.session_state.result = None
    st.session_state.canvas_key = 0

image = Image.open(uploaded_file).convert("RGB")
img_array = np.array(image)

# --- Main Layout ---
tab1, tab2, tab3 = st.tabs(["Exclusions & Calibration", "Results", "Export"])

with tab1:
    st.markdown("### Interactive Canvas")
    st.markdown(
        """
1. Select **point** and click 4 corners of the plotting area.
2. Select **rect** to draw exclusion zones over text/legends.
3. Use canvas toolbar undo/redo, or reset with the button below.
"""
    )

    if st.button("Reset Canvas"):
        st.session_state.canvas_key += 1
        st.session_state.result = None
        st.rerun()

    tool = st.radio("Tool", ["point", "rect", "transform"], horizontal=True)

    canvas_width = 700
    scale_factor = canvas_width / image.width
    canvas_height = int(image.height * scale_factor)

    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.25)",
        stroke_width=2,
        stroke_color="#ff0000",
        background_image=image,
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode=tool,
        point_display_radius=5,
        key=f"canvas_{st.session_state.canvas_key}",
    )

    calibration_points_all = []
    exclusion_zones = []
    objects = []

    if canvas_result.json_data:
        objects = canvas_result.json_data.get("objects", [])
        for obj in objects:
            obj_type = obj.get("type")
            if obj_type == "circle":
                radius = float(obj.get("radius", 0))
                cx = float(obj.get("left", 0)) + radius
                cy = float(obj.get("top", 0)) + radius
                calibration_points_all.append((cx / scale_factor, cy / scale_factor))
            elif obj_type == "rect":
                x0 = float(obj.get("left", 0)) / scale_factor
                y0 = float(obj.get("top", 0)) / scale_factor
                w = float(obj.get("width", 0)) / scale_factor
                h = float(obj.get("height", 0)) / scale_factor
                exclusion_zones.append((x0, y0, x0 + w, y0 + h))

    calibration_points = calibration_points_all[:4]
    ordered_calibration = core.order_calibration_points(calibration_points)

    st.write(f"Calibration Points: {len(calibration_points)} / 4")
    st.write(f"Exclusion Zones: {len(exclusion_zones)}")
    if len(calibration_points_all) > 4:
        st.warning(f"{len(calibration_points_all) - 4} extra point(s) ignored. Keep only 4 corner points.")

    calibration_valid = False
    if len(calibration_points) == 4:
        poly = np.array(ordered_calibration, dtype=np.float32)
        area = cv2.contourArea(poly)
        convex = cv2.isContourConvex(poly.astype(np.int32))
        if area <= 10 or not convex:
            st.warning("Calibration shape looks invalid (non-convex or tiny area). Re-click points.")
        else:
            calibration_valid = True
            st.success("Calibration ready.")
            labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
            corner_df = pd.DataFrame(
                [{"Corner": labels[i], "X(px)": round(p[0], 1), "Y(px)": round(p[1], 1)} for i, p in enumerate(ordered_calibration)]
            )
            st.dataframe(corner_df, hide_index=True, use_container_width=True)
can_extract = len(calibration_points) == 4 and calibration_valid
extract_btn = st.sidebar.button(
    "Extract Data",
    type="primary",
    disabled=not can_extract,
    help="Needs exactly 4 valid calibration points.",
)

if extract_btn:
    with st.spinner("Extracting data..."):
        target_color = ImageColor.getcolor(line_color_hex, "RGB")

        method1_points = []
        method2_points = []
        method1_mask = None
        method2_mask = None

        if run_both_methods or extraction_mode.startswith("Method 1"):
            method1_mask = core.enhance_line_detection(img_array, target_color, color_tolerance)
            method1_points = core.extract_points_method1(
                img_array,
                method1_mask,
                ordered_calibration,
                exclusion_zones=exclusion_zones,
            )

        if run_both_methods or extraction_mode.startswith("Method 2"):
            method2_mask = core.build_method2_mask(
                img_array,
                target_color,
                color_tolerance,
                ordered_calibration,
            )
            core.apply_exclusion_zones(method2_mask, exclusion_zones)
            mode = "count" if sample_mode == "Number of points" else "step"
            method2_points = core.extract_points_method2(
                method2_mask,
                ordered_calibration,
                exclusion_zones=exclusion_zones,
                mode=mode,
                value=int(sample_value),
            )

        method1_real = core.convert_to_real_coordinates(method1_points, ordered_calibration, x_min, x_max, y_min, y_max)
        method2_real = core.convert_to_real_coordinates(method2_points, ordered_calibration, x_min, x_max, y_min, y_max)

        bounds = core.get_calibration_bounds(ordered_calibration, image.width, image.height)
        x_bounds = (bounds[0], bounds[1]) if bounds else (0, image.width)

        m1_quality = _score_points(method1_points, x_bounds)
        m2_quality = _score_points(method2_points, x_bounds)

        active_key = "method1" if extraction_mode.startswith("Method 1") else "method2"
        if active_key == "method1" and not method1_real and method2_real:
            active_key = "method2"
            st.warning("Method 1 found no points. Switched to Method 2 results.")
        if active_key == "method2" and not method2_real and method1_real:
            active_key = "method1"
            st.warning("Method 2 found no points. Switched to Method 1 results.")

        active_real = method1_real if active_key == "method1" else method2_real
        if not active_real:
            st.session_state.result = None
            st.warning("No points detected. Try adjusting color/tolerance or calibration.")
        else:
            active_df = pd.DataFrame(active_real, columns=["X", "Y"])
            st.session_state.result = {
                "active_key": active_key,
                "active_mode": "Method 1" if active_key == "method1" else "Method 2",
                "active_df": active_df,
                "active_stats": core.calculate_stats(active_real),
                "method1": {
                    "points": method1_points,
                    "real": method1_real,
                    "mask": method1_mask,
                    "quality": m1_quality,
                },
                "method2": {
                    "points": method2_points,
                    "real": method2_real,
                    "mask": method2_mask,
                    "quality": m2_quality,
                },
                "image": image,
            }

result = st.session_state.result
if result is not None:
    with tab2:
        st.caption(f"Active extraction method: {result['active_mode']}")

        q1, q2 = st.columns(2)
        q1.metric("Method 1 Score", f"{result['method1']['quality']['score']}")
        q2.metric("Method 2 Score", f"{result['method2']['quality']['score']}")

        compare_df = pd.DataFrame(
            [
                {"Method": "Method 1", **result["method1"]["quality"]},
                {"Method": "Method 2", **result["method2"]["quality"]},
            ]
        )
        st.dataframe(compare_df, hide_index=True, use_container_width=True)

        st.markdown("### Overlay QA")
        overlay_view = st.radio("Overlay View", ["Original", "Mask", "Overlay"], horizontal=True)

        if overlay_view == "Original":
            st.image(result["image"], caption="Original image", use_container_width=True)
        elif overlay_view == "Mask":
            active_mask = result[result["active_key"]]["mask"]
            if active_mask is not None:
                st.image(active_mask, caption=f"{result['active_mode']} mask", use_container_width=True, clamp=True)
            else:
                st.info("No mask available for current method.")
        else:
            overlay_fig = _overlay_figure(
                result["image"],
                m1_points=result["method1"]["points"],
                m2_points=result["method2"]["points"],
                active_points=result[result["active_key"]]["points"],
            )
            st.plotly_chart(overlay_fig, use_container_width=True)

        st.markdown("### Replot")
        rcol1, rcol2, rcol3 = st.columns(3)
        resample_count = rcol1.number_input("Resample points", min_value=2, value=400, step=10)
        smooth_window = rcol2.slider("Smoothing window", min_value=1, max_value=31, value=1, step=2)
        invert_y = rcol3.checkbox("Invert Y axis", value=False)

        replot_df = _prepare_replot_df(result["active_df"], resample_count=resample_count, smooth_window=smooth_window)
        replot_fig = px.line(replot_df, x="X", y="Y", title="Digitized Spectrum")
        if invert_y:
            replot_fig.update_yaxes(autorange="reversed")
        st.plotly_chart(replot_fig, use_container_width=True)
        st.dataframe(replot_df, use_container_width=True)

        st.write("### Statistics")
        st.json(result["active_stats"])

    with tab3:
        csv = result["active_df"].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            csv,
            file_name="digitized_data.csv",
            mime="text/csv",
            key="download-csv",
        )
