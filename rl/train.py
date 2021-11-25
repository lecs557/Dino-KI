"""
Framework to train our Dino model
"""

from .model import DinoRl
from ..io.frame_grabber import FrameGrabber
from ..io.key_trigger import KeyTrigger

fg = FrameGrabber()
kt = KeyTrigger
m = DinoRl()
