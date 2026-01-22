import tkinter as tk
from tkinter import ttk

class MovableStatisticsPanel:
    def __init__(self, parent_app):
        self.parent = parent_app
        self.window = None
        self.is_docked = False
        self.dock_zone_active = False
        self.drag_data = {"x": 0, "y": 0}
        self.dock_indicator = None
        self.dock_tab = None
        self.is_closed = True

    def create_dock_tab(self):
        """Create a small docking tab on the canvas."""
        if self.dock_tab or not hasattr(self.parent, 'canvas'):
            return
            
        try:
            canvas = self.parent.canvas
            # Create a small tab on the left edge
            tab_width, tab_height = 80, 25
            tab_x, tab_y = 10, 100
            
            # Create tab background
            self.dock_tab = canvas.create_rectangle(
                tab_x, tab_y, tab_x + tab_width, tab_y + tab_height,
                fill='#4D6BFE', outline='white', width=2, tags="dock_tab"
            )
            
            # Create tab text
            canvas.create_text(
                tab_x + tab_width//2, tab_y + tab_height//2,
                text="📊 Stats", fill='white', font=('Arial', 9, 'bold'),
                tags="dock_tab"
            )
            
            # Bind click event to show panel
            canvas.tag_bind("dock_tab", "<Button-1>", lambda e: self.show_panel())
            canvas.tag_bind("dock_tab", "<Enter>", self.on_tab_hover)
            canvas.tag_bind("dock_tab", "<Leave>", self.on_tab_leave)
        except Exception as e:
            print(f"Error creating dock tab: {e}")
    
    def remove_dock_tab(self):
        """Remove the docking tab."""
        try:
            if self.dock_tab and hasattr(self.parent, 'canvas'):
                self.parent.canvas.delete("dock_tab")
                self.dock_tab = None
        except Exception as e:
            print(f"Error removing dock tab: {e}")
    
    def on_tab_hover(self, event):
        """Handle tab hover effect."""
        try:
            if hasattr(self.parent, 'canvas'):
                self.parent.canvas.itemconfig("dock_tab", fill='#6C7BFE')
        except Exception as e:
            print(f"Error in tab hover: {e}")
    
    def on_tab_leave(self, event):
        """Handle tab leave effect."""
        try:
            if hasattr(self.parent, 'canvas'):
                self.parent.canvas.itemconfig("dock_tab", fill='#4D6BFE')
        except Exception as e:
            print(f"Error in tab leave: {e}")
        
    def show_panel(self):
        if self.window and self.window.winfo_exists():
            self.window.lift()
            return
        
        self.is_closed = False
        self.remove_dock_tab()  # Remove tab when panel is open
            
        # Create floating statistics window
        self.window = tk.Toplevel(self.parent.root)
        self.window.title("Statistics")
        self.window.geometry("320x450")
        self.window.configure(bg='#f0f0f0')
        self.window.attributes('-alpha', 0.95)  # Slight transparency
        
        # Handle window close event
        self.window.protocol("WM_DELETE_WINDOW", self.hide_panel)
        
        # Make it stay on top but not always
        self.window.attributes('-topmost', False)
        
        # Custom title bar for dragging
        title_bar = tk.Frame(self.window, bg='#4D6BFE', height=30)
        title_bar.pack(fill=tk.X)
        title_bar.pack_propagate(False)
        
        title_label = tk.Label(title_bar, text="📊 Statistics", bg='#4D6BFE', fg='white', 
                              font=('Arial', 10, 'bold'))
        title_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Close button
        close_btn = tk.Button(title_bar, text="✕", bg='#DC3545', fg='white', 
                             font=('Arial', 10, 'bold'), bd=0, width=3,
                             command=self.hide_panel)
        close_btn.pack(side=tk.RIGHT, padx=5, pady=2)
        
        # Dock button
        self.dock_btn = tk.Button(title_bar, text="📌", bg='#28a745', fg='white', 
                                 font=('Arial', 10, 'bold'), bd=0, width=3,
                                 command=self.toggle_dock)
        self.dock_btn.pack(side=tk.RIGHT, padx=2, pady=2)
        
        # Statistics content
        content_frame = tk.Frame(self.window, bg='#f0f0f0')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Text widget with scrollbar
        text_frame = tk.Frame(content_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.stats_text = tk.Text(text_frame, bg='white', wrap=tk.WORD, 
                                 font=('Courier New', 9))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.stats_text.yview)
        self.stats_text.config(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind dragging events
        title_bar.bind("<Button-1>", self.start_drag)
        title_bar.bind("<B1-Motion>", self.on_drag)
        title_bar.bind("<ButtonRelease-1>", self.end_drag)
        title_label.bind("<Button-1>", self.start_drag)
        title_label.bind("<B1-Motion>", self.on_drag)
        title_label.bind("<ButtonRelease-1>", self.end_drag)
        
        # Update content
        self.update_content()
        
        # Position near parent window
        parent_x = self.parent.root.winfo_x()
        parent_y = self.parent.root.winfo_y()
        self.window.geometry(f"320x450+{parent_x + 50}+{parent_y + 50}")
        
    def hide_panel(self):
        if self.window:
            self.window.destroy()
            self.window = None
            self.is_docked = False
            self.is_closed = True
            self.remove_dock_indicator()
            self.create_dock_tab()  # Create tab when panel is closed
            
    def start_drag(self, event):
        self.drag_data["x"] = event.x_root
        self.drag_data["y"] = event.y_root
        
    def on_drag(self, event):
        if not self.window:
            return
            
        # Calculate movement
        dx = event.x_root - self.drag_data["x"]
        dy = event.y_root - self.drag_data["y"]
        
        # Move window
        x = self.window.winfo_x() + dx
        y = self.window.winfo_y() + dy
        self.window.geometry(f"+{x}+{y}")
        
        # Check for dock zone (left side of parent window)
        parent_x = self.parent.root.winfo_x()
        parent_y = self.parent.root.winfo_y()
        parent_width = self.parent.root.winfo_width()
        parent_height = self.parent.root.winfo_height()
        
        dock_zone_x = parent_x + 10
        dock_zone_width = 100
        
        if (dock_zone_x <= event.x_root <= dock_zone_x + dock_zone_width and
            parent_y <= event.y_root <= parent_y + parent_height):
            self.show_dock_indicator()
        else:
            self.remove_dock_indicator()
        
        self.drag_data["x"] = event.x_root
        self.drag_data["y"] = event.y_root
        
    def end_drag(self, event):
        # Check if we should dock
        parent_x = self.parent.root.winfo_x()
        parent_y = self.parent.root.winfo_y()
        parent_height = self.parent.root.winfo_height()
        
        dock_zone_x = parent_x + 10
        dock_zone_width = 100
        
        if (dock_zone_x <= event.x_root <= dock_zone_x + dock_zone_width and
            parent_y <= event.y_root <= parent_y + parent_height):
            self.dock_to_parent()
        
        self.remove_dock_indicator()
        
    def show_dock_indicator(self):
        if not self.dock_indicator and self.parent.root.winfo_exists():
            # Create a visual indicator on the main canvas
            canvas = self.parent.canvas
            canvas_width = canvas.winfo_width()
            
            self.dock_indicator = canvas.create_rectangle(
                5, 5, 325, canvas.winfo_height() - 5,
                outline='#4D6BFE', width=3, dash=(10, 5),
                tags="dock_indicator"
            )
            
    def remove_dock_indicator(self):
        if self.dock_indicator:
            self.parent.canvas.delete("dock_indicator")
            self.dock_indicator = None
            
    def dock_to_parent(self):
        if not self.window:
            return
            
        parent_x = self.parent.root.winfo_x()
        parent_y = self.parent.root.winfo_y()
        
        # Position at left side of parent
        self.window.geometry(f"320x450+{parent_x + 10}+{parent_y + 80}")
        self.is_docked = True
        self.dock_btn.config(text="📌", bg='#ffc107')  # Change icon when docked
        
    def toggle_dock(self):
        if self.is_docked:
            # Undock - move away from parent
            parent_x = self.parent.root.winfo_x()
            parent_y = self.parent.root.winfo_y()
            self.window.geometry(f"320x450+{parent_x + 400}+{parent_y + 100}")
            self.is_docked = False
            self.dock_btn.config(text="📌", bg='#28a745')
        else:
            self.dock_to_parent()
            
    def update_content(self):
        if not self.window or not self.window.winfo_exists() or not hasattr(self, 'stats_text'):
            return
            
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        
        if not self.parent.statistics:
            self.stats_text.insert(tk.END, "No data available for statistics.\n\nExtract data first to see:\n• Point counts\n• Statistical measures\n• Best fit analysis\n• Extrema detection")
            self.stats_text.config(state=tk.DISABLED)
            return
            
        stats_str = "--- Line Masking Results ---
"
        if hasattr(self.parent, 'line_masking') and self.parent.line_masking.detected_lines:
            stats_str += f"Lines Detected: {len(self.parent.line_masking.detected_lines)}\n"
            stats_str += f"Current Line: {self.parent.line_masking.current_line_index + 1}\n\n"
        
        stats_str += "--- Method Results ---
"
        stats_str += f"Method 1: {len(self.parent.method1_points)} points\n"
        stats_str += f"Method 2: {len(self.parent.method2_points)} points\n"
        stats_str += f"Current: {self.parent.method_var.get()}\n\n"
        
        stats_str += "--- General ---
"
        stats_str += f"{'.':<15} {self.parent.statistics['Points']}\n"
        stats_str += f"{'.':<15} {self.parent.statistics.get('Correlation', 0):.4f}\n"
        stats_str += f"{'.':<15} {self.parent.statistics.get('Area (Trapezoid)', 0):.4f}\n\n"
        
        if 'Peak' in self.parent.statistics:
            stats_str += "--- Extrema ---
"
            peak = self.parent.statistics['Peak']
            valley = self.parent.statistics['Valley']
            stats_str += f"{'.':<15} ({peak[0]:.2f}, {peak[1]:.2f})\n"
            stats_str += f"{'.':<15} ({valley[0]:.2f}, {valley[1]:.2f})\n\n"
            
        stats_str += "--- X-Axis Stats ---
"
        stats_str += f"{'.':<15} {self.parent.statistics.get('X Mean', 0):.4f}\n"
        stats_str += f"{'.':<15} {self.parent.statistics.get('X Std Dev', 0):.4f}\n"
        stats_str += f"{'.':<15} {self.parent.statistics.get('X Skew', 0):.4f}\n"
        stats_str += f"{'.':<15} {self.parent.statistics.get('X Kurtosis', 0):.4f}\n\n"
        
        stats_str += "--- Y-Axis Stats ---
"
        stats_str += f"{'.':<15} {self.parent.statistics.get('Y Mean', 0):.4f}\n"
        stats_str += f"{'.':<15} {self.parent.statistics.get('Y Std Dev', 0):.4f}\n"
        stats_str += f"{'.':<15} {self.parent.statistics.get('Y Skew', 0):.4f}\n"
        stats_str += f"{'.':<15} {self.parent.statistics.get('Y Kurtosis', 0):.4f}\n\n"
        
        if self.parent.best_fit_equation:
            stats_str += "--- Best Fit Line ---
"
            stats_str += f"{'.':<15} {self.parent.best_fit_equation}\n"
            stats_str += f"{'.':<15} {self.parent.statistics.get('r_squared', 0):.6f}
