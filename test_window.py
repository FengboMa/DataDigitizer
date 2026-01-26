import tkinter as tk
import time
import sys

print("Starting Tkinter test...", flush=True)
try:
    root = tk.Tk()
    root.title("Test Window")
    root.geometry("200x200")
    label = tk.Label(root, text="Hello!")
    label.pack()
    print("Window created. Updating...", flush=True)
    root.update()
    print("Window updated. Sleeping 2s...", flush=True)
    time.sleep(2)
    print("Destroying...", flush=True)
    root.destroy()
    print("Done.", flush=True)
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
