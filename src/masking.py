import tkinter as tk
from tkinter import messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk

class LineMaskingSystem:
    def __init__(self, parent_app):
        self.parent = parent_app
        self.detected_lines = []  # List of line masks
        self.current_line_index = 0
        self.line_colors = []  # Colors for visualization
        self.line_previews = []  # Thumbnail previews
        self.detection_parameters = {
            'min_line_length': 50,
            'max_line_gap': 10,
            'thickness_tolerance': 5,
            'clustering_eps': 15,
            'min_samples': 10
        }
        
    def detect_all_lines(self):
        """
        Advanced line detection using multiple computer vision techniques.
        Detects and separates individual spectral lines in complex multi-line plots.
        Now respects exclusion zones to avoid detecting text and labels.
        """
        if self.parent.image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return
        
        try:
            # Update debug display
            self.parent.debug_label.config(text="Debug: Starting detection...")
            self.parent.root.update_idletasks()
            
            print("Starting advanced line detection with exclusion zones...")
            
            # Get parameters from UI
            edge_thresh = int(self.parent.edge_threshold_var.get())
            min_length = int(self.parent.min_length_var.get())
            cluster_dist = int(self.parent.cluster_dist_var.get())
            min_points = int(self.parent.min_points_var.get())
            
            # Update detection parameters
            self.detection_parameters.update({
                'min_line_length': min_length,
                'clustering_eps': cluster_dist,
                'min_samples': min_points
            })
            
            # Convert to working formats
            if self.parent.original_image is not None:
                img_array = np.array(self.parent.original_image)
            else:
                img_array = np.array(self.parent.image)
            
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # NEW: Create exclusion mask to avoid detecting text/labels
            exclusion_mask = self._create_exclusion_mask(gray.shape)
            if exclusion_mask is not None:
                self.parent.debug_label.config(text=f"Debug: Applying {len(self.parent.exclusion_zones)} exclusion zones...")
                self.parent.root.update_idletasks()
                print(f"Applying {len(self.parent.exclusion_zones)} exclusion zones to line detection")
            
            # Clear previous results
            self.detected_lines.clear()
            self.line_colors.clear()
            self.line_previews.clear()
            
            # Method 1: Edge-based line detection with Hough Transform
            self.parent.debug_label.config(text="Debug: Running Hough detection...")
            self.parent.root.update_idletasks()
            lines_hough = self._detect_lines_hough(gray, edge_thresh, exclusion_mask)
            print(f"Hough detection found {len(lines_hough)} points")
            
            # Method 2: Contour-based line detection
            self.parent.debug_label.config(text="Debug: Running contour detection...")
            self.parent.root.update_idletasks()
            lines_contour = self._detect_lines_contour(gray, exclusion_mask)
            print(f"Contour detection found {len(lines_contour)} points")
            
            # Method 3: Skeletonization approach for complex overlapping lines
            self.parent.debug_label.config(text="Debug: Running skeleton detection...")
            self.parent.root.update_idletasks()
            lines_skeleton = self._detect_lines_skeleton(gray, exclusion_mask)
            print(f"Skeleton detection found {len(lines_skeleton)} points")
            
            # Combine and separate overlapping detections
            all_line_points = lines_hough + lines_contour + lines_skeleton
            print(f"Total points before clustering: {len(all_line_points)}")
            
            if not all_line_points:
                self.parent.debug_label.config(text="Debug: No points found. Try adjusting parameters.")
                messagebox.showinfo("No Lines", "No distinct lines detected. Try:\n1. Lower edge threshold\n2. Smaller min length\n3. Use Simple Detection button\n4. First run Auto-Exclude Text if not done")
                return
            
            # Cluster points into separate lines using custom clustering
            self.parent.debug_label.config(text="Debug: Clustering points into lines...")
            self.parent.root.update_idletasks()
            separated_lines = self._separate_lines_clustering(all_line_points)
            print(f"Clustering produced {len(separated_lines)} lines")
            
            if not separated_lines:
                self.parent.debug_label.config(text="Debug: Clustering failed. Try larger cluster distance.")
                messagebox.showinfo("No Lines", "Clustering failed to separate lines. Try:\n1. Increase cluster distance\n2. Decrease min points\n3. Use Simple Detection")
                return
            
            # Create masks for each detected line
            self.parent.debug_label.config(text="Debug: Creating line masks...")
            self.parent.root.update_idletasks()
            self._create_line_masks(separated_lines, gray.shape)
            
            # Generate colors and previews for UI
            self._generate_line_visualization()
            
            self.parent.debug_label.config(text=f"Debug: Success! Found {len(self.detected_lines)} lines")
            print(f"Successfully detected {len(self.detected_lines)} distinct lines")
            messagebox.showinfo("Line Detection Complete", 
                              f"Detected {len(self.detected_lines)} distinct lines.\n"
                              f"Use the controls to cycle through and select lines for extraction.")
            
        except Exception as e:
            self.parent.debug_label.config(text=f"Debug: Error - {str(e)[:50]}...")
            messagebox.showerror("Line Detection Error", f"Failed to detect lines: {e}")
            print(f"Line detection error: {e}")
    
    def _create_exclusion_mask(self, img_shape):
        """Create a mask that excludes regions defined in exclusion_zones."""
        if not self.parent.exclusion_zones:
            return None
        
        # Create mask where excluded regions are 0, allowed regions are 255
        mask = np.ones(img_shape, dtype=np.uint8) * 255
        
        for (x0, y0, x1, y1) in self.parent.exclusion_zones:
            # Ensure coordinates are within image bounds
            x0 = max(0, min(x0, img_shape[1]-1))
            y0 = max(0, min(y0, img_shape[0]-1))
            x1 = max(0, min(x1, img_shape[1]-1))
            y1 = max(0, min(y1, img_shape[0]-1))
            
            # Set excluded region to 0
            mask[y0:y1+1, x0:x1+1] = 0
        
        return mask
    
    def _detect_lines_hough(self, gray, edge_thresh=30, exclusion_mask=None):
        """Use Hough Line Transform to detect straight line segments, respecting exclusion zones."""
        # Enhanced edge detection with adjustable threshold
        edges = cv2.Canny(gray, edge_thresh, edge_thresh * 3, apertureSize=3)
        
        # Apply exclusion mask if provided
        if exclusion_mask is not None:
            edges = cv2.bitwise_and(edges, exclusion_mask)
        
        # Apply HoughLinesP for line segment detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                               threshold=max(15, edge_thresh//2),
                               minLineLength=self.detection_parameters['min_line_length'],
                               maxLineGap=self.detection_parameters['max_line_gap'])
        
        line_points = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Sample points along each detected line
                num_points = max(5, int(np.sqrt((x2-x1)**2 + (y2-y1)**2) / 3))
                x_points = np.linspace(x1, x2, num_points, dtype=int)
                y_points = np.linspace(y1, y2, num_points, dtype=int)
                
                for x, y in zip(x_points, y_points):
                    if (0 <= x < gray.shape[1] and 0 <= y < gray.shape[0] and
                        (exclusion_mask is None or exclusion_mask[y, x] > 0)):
                        line_points.append((x, y))
        
        return line_points
    
    def _detect_lines_contour(self, gray, exclusion_mask=None):
        """Use contour detection to find curved and irregular lines, respecting exclusion zones."""
        # Adaptive thresholding to handle varying intensities
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Apply exclusion mask if provided
        if exclusion_mask is not None:
            binary = cv2.bitwise_and(binary, exclusion_mask)
        
        # Morphological operations to clean up the binary image
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        line_points = []
        for contour in contours:
            # Filter contours based on area and aspect ratio
            area = cv2.contourArea(contour)
            if area < 100:  # Too small
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Look for line-like shapes (high aspect ratio or reasonable area)
            if aspect_ratio > 3 or (area > 200 and aspect_ratio > 1.5):
                # Sample points along the contour
                contour_points = contour.reshape(-1, 2)
                for point in contour_points[::2]:  # Sample every other point
                    x, y = tuple(point)
                    if (exclusion_mask is None or 
                        (0 <= y < exclusion_mask.shape[0] and 0 <= x < exclusion_mask.shape[1] and exclusion_mask[y, x] > 0)):
                        line_points.append((x, y))
        
        return line_points
    
    def _detect_lines_skeleton(self, gray, exclusion_mask=None):
        """Use morphological skeletonization to detect line centers, respecting exclusion zones."""
        # Binary threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Apply exclusion mask if provided
        if exclusion_mask is not None:
            binary = cv2.bitwise_and(binary, exclusion_mask)
        
        # Skeletonization using morphological operations
        skeleton = np.zeros_like(binary)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        
        while True:
            eroded = cv2.erode(binary, kernel)
            dilated = cv2.dilate(eroded, kernel)
            skeleton = cv2.bitwise_or(skeleton, cv2.subtract(binary, dilated))
            binary = eroded.copy()
            
            if cv2.countNonZero(binary) == 0:
                break
        
        # Extract skeleton points
        line_points = []
        y_coords, x_coords = np.where(skeleton > 0)
        for x, y in zip(x_coords, y_coords):
            if (exclusion_mask is None or 
                (0 <= y < exclusion_mask.shape[0] and 0 <= x < exclusion_mask.shape[1] and exclusion_mask[y, x] > 0)):
                line_points.append((x, y))
        
        return line_points
    
    def _separate_lines_clustering(self, all_points):
        """Use simple distance-based clustering to separate individual lines."""
        if not all_points:
            return []
        
        # Convert to numpy array for easier processing
        points_array = np.array(all_points)
        
        # Simple distance-based clustering
        eps = self.detection_parameters['clustering_eps']
        min_samples = self.detection_parameters['min_samples']
        
        clusters = []
        visited = np.zeros(len(points_array), dtype=bool)
        
        for i, point in enumerate(points_array):
            if visited[i]:
                continue
                
            # Find all points within eps distance
            distances = np.sqrt(np.sum((points_array - point) ** 2, axis=1))
            neighbors = np.where(distances <= eps)[0]
            
            if len(neighbors) < min_samples:
                continue  # Skip noise points
            
            # Create new cluster
            cluster = []
            queue = list(neighbors)
            
            while queue:
                current_idx = queue.pop(0)
                if visited[current_idx]:
                    continue
                    
                visited[current_idx] = True
                cluster.append(points_array[current_idx])
                
                # Find neighbors of current point
                current_point = points_array[current_idx]
                distances = np.sqrt(np.sum((points_array - current_point) ** 2, axis=1))
                current_neighbors = np.where(distances <= eps)[0]
                
                # Add unvisited neighbors to queue
                for neighbor_idx in current_neighbors:
                    if not visited[neighbor_idx]:
                        queue.append(neighbor_idx)
            
            if len(cluster) >= min_samples:
                clusters.append(np.array(cluster))
        
        # Further separate by Y-coordinate proximity (for horizontal lines)
        separated_lines = []
        for cluster_points in clusters:
            if len(cluster_points) == 0:
                continue
                
            # Sort by Y coordinate
            y_sorted = cluster_points[cluster_points[:, 1].argsort()]
            
            # Split into sub-lines if there are large Y gaps
            sub_lines = []
            current_line = [y_sorted[0]]
            
            for i in range(1, len(y_sorted)):
                if abs(y_sorted[i][1] - y_sorted[i-1][1]) < 20:  # Same line
                    current_line.append(y_sorted[i])
                else:  # New line
                    if len(current_line) > 20:  # Minimum points for a valid line
                        sub_lines.append(np.array(current_line))
                    current_line = [y_sorted[i]]
            
            if len(current_line) > 20:
                sub_lines.append(np.array(current_line))
            
            separated_lines.extend(sub_lines)
        
        return separated_lines
    
    def _create_line_masks(self, separated_lines, img_shape):
        """Create binary masks for each detected line."""
        for i, line_points in enumerate(separated_lines):
            if len(line_points) < 10:
                continue
            
            # Create binary mask
            mask = np.zeros(img_shape, dtype=np.uint8)
            
            # Draw thick line through all points
            for point in line_points:
                x, y = point
                if 0 <= x < img_shape[1] and 0 <= y < img_shape[0]:
                    # Draw a small circle at each point to create thickness
                    cv2.circle(mask, (int(x), int(y)), 
                             self.detection_parameters['thickness_tolerance'], 255, -1)
            
            # Smooth the mask
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            self.detected_lines.append(mask)
    
    def _generate_line_visualization(self):
        """Generate colors and preview thumbnails for each detected line."""
        # Generate distinct colors for each line
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, mask in enumerate(self.detected_lines):
            # Assign color
            color = colors[i % len(colors)]
            self.line_colors.append(color)
            
            # Create preview thumbnail
            try:
                preview = self._create_line_preview(mask, color)
                self.line_previews.append(preview)
            except Exception as e:
                print(f"Failed to create preview for line {i}: {e}")
                # Add a None placeholder so indices stay aligned
                self.line_previews.append(None)
    
    def _create_line_preview(self, mask, color):
        """Create a small preview image of the line for UI display."""
        # Find bounding box of the line
        y_coords, x_coords = np.where(mask > 0)
        if len(x_coords) == 0:
            return None
        
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        
        # Ensure we have a reasonable bounding box
        if max_x - min_x < 5 or max_y - min_y < 2:
            # Create a simple line preview if bounding box is too small
            preview_rgb = np.zeros((30, 100, 3), dtype=np.uint8)
            color_map = {'red': (255,0,0), 'blue': (0,0,255), 'green': (0,255,0), 
                        'orange': (255,165,0), 'purple': (128,0,128), 'brown': (165,42,42), 
                        'pink': (255,192,203), 'gray': (128,128,128)}
            rgb_color = color_map.get(color, (255,255,255))
            
            # Draw a simple horizontal line
            preview_rgb[12:18, 10:90] = rgb_color
            return Image.fromarray(preview_rgb)
        
        # Extract region and resize for preview
        try:
            line_region = mask[min_y:max_y+1, min_x:max_x+1]
            
            # Resize to thumbnail size
            thumbnail_size = (100, 30)
            line_region_resized = cv2.resize(line_region, thumbnail_size)
            
            # Convert to RGB and apply color
            preview_rgb = np.zeros((*thumbnail_size[::-1], 3), dtype=np.uint8)
            color_map = {'red': (255,0,0), 'blue': (0,0,255), 'green': (0,255,0), 
                        'orange': (255,165,0), 'purple': (128,0,128), 'brown': (165,42,42), 
                        'pink': (255,192,203), 'gray': (128,128,128)}
            rgb_color = color_map.get(color, (255,255,255))
            for c in range(3):
                preview_rgb[:,:,c] = (line_region_resized / 255) * rgb_color[c]
            
            return Image.fromarray(preview_rgb)
        except Exception as e:
            print(f"Error creating detailed preview: {e}")
            # Fallback to simple line preview
            preview_rgb = np.zeros((30, 100, 3), dtype=np.uint8)
            color_map = {'red': (255,0,0), 'blue': (0,0,255), 'green': (0,255,0), 
                        'orange': (255,165,0), 'purple': (128,0,128), 'brown': (165,42,42), 
                        'pink': (255,192,203), 'gray': (128,128,128)}
            rgb_color = color_map.get(color, (255,255,255))
            preview_rgb[12:18, 10:90] = rgb_color
            return Image.fromarray(preview_rgb)
    
    def get_current_line_mask(self):
        """Get the currently selected line mask."""
        if not self.detected_lines or self.current_line_index >= len(self.detected_lines):
            return None
        return self.detected_lines[self.current_line_index]
    
    def next_line(self):
        """Switch to the next detected line."""
        if self.detected_lines:
            self.current_line_index = (self.current_line_index + 1) % len(self.detected_lines)
            return True
        return False
    
    def previous_line(self):
        """Switch to the previous detected line."""
        if self.detected_lines:
            self.current_line_index = (self.current_line_index - 1) % len(self.detected_lines)
            return True
        return False
    
    def set_line_index(self, index):
        """Set the current line index directly."""
        if 0 <= index < len(self.detected_lines):
            self.current_line_index = index
            return True
        return False
