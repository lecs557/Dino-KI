from mss import mss
import matplotlib.patches as patches
import matplotlib.pyplot as pl
import numpy as np

frame_coords = (0, 0, 1690, 90)  # frame (left, upper, right, lower)
dino_coords = (235, 236, 100, 450)  # dino field of view (up, lo, l, r)
score_coords = (10, 40, 1050, 1175)  # score location (up, lo, l, r)
gameover_coords = (150, 210, 540, 610)  # Game Over location (up, lo, l, r)

with mss() as sct:
    screen = sct.grab(monitor=frame_coords)

frame = np.array(screen)
frame = np.mean(frame, axis=-1)

dino_frame = frame[dino_coords[0]:dino_coords[1],
             dino_coords[2]:dino_coords[3]
             ]

score_frame = frame[score_coords[0]: score_coords[1],
              score_coords[2]:score_coords[3]]

gameover_frame = frame[gameover_coords[0]: gameover_coords[1],
                 gameover_coords[2]:gameover_coords[3]]

pl.imshow(frame, cmap="Greys_r")

rect = patches.Rectangle((dino_coords[2], dino_coords[0]),
                         dino_coords[3] - dino_coords[2],
                         dino_coords[1] - dino_coords[0],
                         linewidth=1, edgecolor='r',
                         facecolor='none', alpha=0.5)

rect = patches.Rectangle((score_coords[2], score_coords[0]),
                         score_coords[3] - score_coords[2],
                         score_coords[1] - score_coords[0],
                         linewidth=1, edgecolor='r',
                         facecolor='none', alpha=0.5)

rect = patches.Rectangle((gameover_coords[2], gameover_coords[0]),
                         gameover_coords[3] - gameover_coords[2],
                         gameover_coords[1] - gameover_coords[0],
                         linewidth=1, edgecolor='r',
                         facecolor='none', alpha=0.5)
