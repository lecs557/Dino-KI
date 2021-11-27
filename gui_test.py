import tkinter as tk
from tkinter import ttk
from mss import mss
import numpy as np
from PIL import Image, ImageTk

frame_coords = (0, 70, 1600, 750)  # frame (left, upper, right, lower)
dino_coords = (235, 236, 100, 450)  # dino field of view (up, lo, l, r)
score_coords = (10, 40, 1050, 1175)  # score location (up, lo, l, r)
gameover_coords = (150, 210, 540, 610)  # Game Over location (up, lo, l, r)

with mss() as sct:
    screen = sct.grab(monitor=frame_coords)

frame = np.array(screen)
frame = np.mean(frame, axis=-1)

root = tk.Tk()
image = Image.fromarray(frame)
img = ImageTk.PhotoImage(image)
frm = ttk.Frame(root, padding=10)
#frm.grid()
#ttk.Label(frm, text="Wer kapiert macht das denn?").grid(column=0, row=0)
#ttk.Label(frm, text="Hello World!").grid(column=0, row=1)
#ttk.Button(frm, text="Quit", command=root.destroy).grid(column=1, row=1)
frm.pack()
ttk.Label(frm, text="Wer macht das denn?").pack()
ttk.Label(frm, text="Hello World!", image=img).pack()
ttk.Button(frm, text="Quit", command=root.destroy).pack()
root.mainloop()