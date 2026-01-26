import sys
print(f"Python version: {sys.version}")
try:
    import tkinter
    print(f"Tkinter version: {tkinter.TkVersion}")
    root = tkinter.Tk()
    print("Tkinter initialized successfully")
    root.destroy()
except Exception as e:
    print(f"Tkinter error: {e}")
