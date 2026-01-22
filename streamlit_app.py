import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageColor
import cv2
from streamlit_drawable_canvas import st_canvas
import plotly.express as px
from src import core

st.set_page_config(page_title="Raman Data Digitizer", layout="wide")

st.title("📊 Raman Data Digitizer (Web)")

# --- Sidebar Configuration ---
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
    extraction_mode = st.selectbox("Method", ["Method 1 (Color Mask)", "Method 2 (Advanced)"])
    
    extract_btn = st.button("🚀 Extract Data", type="primary")

if not uploaded_file:
    st.info("Please upload an image to start.")
    st.stop()

# Load Image
image = Image.open(uploaded_file).convert("RGB")
img_array = np.array(image)

# --- Main Layout ---
tab1, tab2, tab3 = st.tabs(["🛑 Exclusions & Calibration", "🎯 Results", "📥 Export"])

with tab1:
    st.markdown("### Interactive Canvas")
    st.markdown("""
    1. **Calibration:** Select 'Point' tool. Click 4 corners (Top-Left, Top-Right, Bottom-Right, Bottom-Left).
    2. **Exclusion:** Select 'Rect' tool. Draw boxes around text/legends to ignore.
    """)
    
    # Canvas Controls
    tool = st.radio("Tool:", ["point", "rect", "transform"], horizontal=True)
    
    # Canvas
    # We use a fixed width/height for display but map coordinates back to original image
    canvas_width = 700
    scale_factor = canvas_width / image.width
    canvas_height = int(image.height * scale_factor)
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=2,
        stroke_color="#ff0000",
        background_image=image,
        update_streamlit=True,
        height=canvas_height,
        width=canvas_width,
        drawing_mode=tool,
        point_display_radius=5,
        key="canvas",
    )

    # Process Canvas Inputs
    calibration_points = []
    exclusion_zones = []
    
    if canvas_result.json_data:
        objects = canvas_result.json_data["objects"]
        for obj in objects:
            if obj["type"] == "circle": # Point
                # Convert canvas coords to original image coords
                cx = obj["left"] + obj["radius"]
                cy = obj["top"] + obj["radius"]
                calibration_points.append((cx / scale_factor, cy / scale_factor))
            elif obj["type"] == "rect": # Exclusion Zone
                x = obj["left"] / scale_factor
                y = obj["top"] / scale_factor
                w = obj["width"] / scale_factor
                h = obj["height"] / scale_factor
                exclusion_zones.append((x, y, x+w, y+h))

    st.write(f"**Calibration Points:** {len(calibration_points)} / 4")
    if len(calibration_points) == 4:
        st.success("Calibration Ready!")
    elif len(calibration_points) > 4:
        st.warning("Too many points! Please undo or clear.")

# --- Extraction Logic ---
if extract_btn:
    if len(calibration_points) != 4:
        st.error("Please select exactly 4 calibration points first.")
    else:
        with st.spinner("Extracting data..."):
            # Prepare data
            target_color = ImageColor.getcolor(line_color_hex, "RGB")
            
            # Sort calibration points (TL, TR, BR, BL)
            # Simple sorting assumption: Sort by Y (Top/Bottom), then by X (Left/Right)
            # This logic assumes a roughly rectangular layout.
            sorted_pts = sorted(calibration_points, key=lambda p: p[1])
            top = sorted(sorted_pts[:2], key=lambda p: p[0])
            bottom = sorted(sorted_pts[2:], key=lambda p: p[0])
            # Order: TL, TR, BR, BL (Core logic expects perspective src points)
            # Wait, src/core expects src points.
            # src/app.py logic was: src = [TL, TR, BR, BL]
            calib_final = [top[0], top[1], bottom[1], bottom[0]]
            
            # Create Line Mask
            # Method 1 Logic
            mask = core.enhance_line_detection(img_array, target_color, color_tolerance)
            
            # Apply Exclusion Zones to Mask
            for (ex, ey, ex2, ey2) in exclusion_zones:
                cv2.rectangle(mask, (int(ex), int(ey)), (int(ex2), int(ey2)), 0, -1)
            
            # Extract Points (Simplified Method 1 approach for now)
            gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            extracted_points = []
            
            min_x_px = int(min(p[0] for p in calib_final))
            max_x_px = int(max(p[0] for p in calib_final))
            min_y_px = int(min(p[1] for p in calib_final))
            max_y_px = int(max(p[1] for p in calib_final))
            
            for x in range(min_x_px, max_x_px):
                if x >= mask.shape[1]: continue
                col = mask[min_y_px:max_y_px, x]
                y_indices = np.where(col > 0)[0]
                
                if y_indices.size > 0:
                    # Weighted average for sub-pixel accuracy
                    y_coords = y_indices + min_y_px
                    weights = 255.0 - gray_image[y_coords, x].astype(float)
                    if np.sum(weights) > 0:
                        y_center = np.sum(y_coords * weights) / np.sum(weights)
                        extracted_points.append((float(x), y_center))
                    else:
                        extracted_points.append((float(x), np.mean(y_coords)))
            
            # Convert to Real Coordinates
            if extracted_points:
                real_data = core.convert_to_real_coordinates(
                    extracted_points, calib_final, x_min, x_max, y_min, y_max
                )
                
                # Display Results
                df = pd.DataFrame(real_data, columns=["X", "Y"])
                
                with tab2:
                    st.plotly_chart(px.line(df, x="X", y="Y", title="Digitized Spectrum"), use_container_width=True)
                    st.dataframe(df)
                    
                    # Stats
                    stats = core.calculate_stats(real_data)
                    st.write("### Statistics")
                    st.json(stats)
                
                with tab3:
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download CSV",
                        csv,
                        "digitized_data.csv",
                        "text/csv",
                        key='download-csv'
                    )
                
                # Auto-switch to Results tab
                # st.experimental_set_query_params() # Hack to refresh, but not needed
                
            else:
                st.warning("No points detected. Try adjusting the Color Tolerance or picking a better color.")

