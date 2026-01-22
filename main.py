import tkinter as tk
from src.app import RamanDataDigitizer

if __name__ == "__main__":
    # Create the main tkinter window
    root = tk.Tk()
    
    # Create and run the Enhanced Raman Data Digitizer application
    app = RamanDataDigitizer(root)
    
    # Start the tkinter event loop
    root.mainloop()
