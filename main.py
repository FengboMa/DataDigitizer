import tkinter as tk
import sys
import traceback

try:
    print("Starting main...", flush=True)
    from src.app import RamanDataDigitizer
    print("Imported RamanDataDigitizer...", flush=True)

    if __name__ == "__main__":
        print("Creating Tk root...", flush=True)
        # Create the main tkinter window
        root = tk.Tk()
        
        print("Initializing app...", flush=True)
        # Create and run the Enhanced Raman Data Digitizer application
        app = RamanDataDigitizer(root)
        
        print("Starting mainloop...", flush=True)
        # Start the tkinter event loop
        root.mainloop()
        print("Mainloop finished.", flush=True)
except Exception as e:
    print("Caught exception:", file=sys.stderr)
    traceback.print_exc()