#Running The GUI interface
import tkinter as tk
from running_module import Application

root = tk.Tk()
app = Application(master=root)
app.mainloop()
