import tkinter as tk
from tkinter import ttk
from mss import mss
import numpy as np
from PIL import Image, ImageTk

frame_coords = (0, 0, 500, 500)  # frame (left, upper, right, lower)

def pil_frombytes(im):
    """ Efficient Pillow version. """
    return Image.frombytes('RGB', im.size, im.bgra, 'raw', 'BGRX')

#screenshot
with mss() as sct:
    screen = sct.grab(monitor=frame_coords)

img1 = Image.new("RGB", screen.size)
pixels = zip(screen.raw[2::4], screen.raw[1::4], screen.raw[0::4])
img1.putdata(list(pixels))

img2 = pil_frombytes(screen)

array = np.array(screen)
# RGBA vertauscht mit BGRA
array[:,:,[0,1,2,3]] = array[:,:,[2,1,0,3]]
array[:,:,3]=255
img3 = Image.fromarray(array)
img3.show()
