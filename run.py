"""
Dino KI
Inspiration von
https://learndatasci.com/tutorials/reinforcement-q-learning-scratch-python
-openai-gym/
"""

import sys
from time import sleep

import matplotlib.patches as patches
import matplotlib.pyplot as pl
import numpy as np
import pyautogui
import pytesseract
from mss import mss


def get_distance(a):
    max_val = a.max()
    a_red = np.min(a, axis=0)
    ds = np.where(a_red < max_val)[0]
    if len(ds) > 0:
        return int(num_distance_buckets * ds[0] / a_red.shape[0])
    else:
        return num_distance_buckets - 1  # a_red.shape[0] - 1


def get_score(frame):
    res = pytesseract.image_to_string(frame)
    try:
        return int(res.split(" ")[-1])
    except ValueError:
        return -1


def get_gameover(frame):
    return frame.mean() < 90


def push_states():
    last_states.append(distance)
    last_states.append(new_distance)

    while len(last_states) > num_historic_states:
        del last_states[0]


def get_screenshots():
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

    return frame, dino_frame, score_frame, gameover_frame


test = {"top": 40, "left": 0, "width": 800, "height": 640}
frame_coords = (110, 250, 1290, 550)  # frame (left, upper, right, lower)
dino_coords = (235, 236, 100, 450)  # dino field of view (up, lo, l, r)
score_coords = (10, 40, 1050, 1175)  # score location (up, lo, l, r)
gameover_coords = (150, 210, 540, 610)  # Game Over location (up, lo, l, r)

num_epochs = 10000
plotting = True

if plotting:
    fig, axs = pl.subplots(3, 1, figsize=(100, 100))

num_distance_buckets = 10
num_historic_states = 10
q_table = np.zeros([num_distance_buckets, 2])
q_table[:, 0] = 0.1  # slight preference for "not jumping"

# hyper parameters
alpha = 0.1
gamma = 0.6
epsilon = 0.

# first iteration
frame, dino_frame, score_frame, gameover_frame = get_screenshots()

score = get_score(score_frame)
distance = get_distance(dino_frame)

last_states = []
last_actions = []

for ep in range(num_epochs):
    if np.random.rand() < epsilon:
        # Explore action space
        action = np.random.randint(2)
        print("picked random action", action)
    else:
        # Exploit learned values
        action = np.argmax(q_table[distance])
        print("use learned action", action)

    if action:
        pyautogui.press("up")
        sleep(0.2)

    frame, dino_frame, score_frame, gameover_frame = get_screenshots()

    new_score = get_score(score_frame)
    new_distance = get_distance(dino_frame)
    is_gameover = get_gameover(gameover_frame)

    push_states()

    if is_gameover:
        # we hit a cactus
        reward = -10
        pyautogui.press("up")
        print("================== GAME OVER ==========")
        sleep(3)  # wait for the Game Over sign to go away
    else:
        reward = 2

    old_value = q_table[distance, action]
    next_max = np.max(q_table[new_distance])

    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

    if is_gameover:
        for s in last_states:
            if not s == num_distance_buckets - 1:
                # when there is nothing on the radar, don't update
                q_table[s, action] = new_value
                print("assigning", s, q_table[s])
    else:
        q_table[distance, action] = new_value

    distance = new_distance
    score = new_score

    if plotting:
        axs[0].imshow(frame, cmap="Greys_r")
        rect = patches.Rectangle((dino_coords[2], dino_coords[0]),
                                 dino_coords[3] - dino_coords[2],
                                 dino_coords[1] - dino_coords[0],
                                 linewidth=1, edgecolor='r',
                                 facecolor='none', alpha=0.5)
        axs[0].add_patch(rect)

        rect = patches.Rectangle((score_coords[2], score_coords[0]),
                                 score_coords[3] - score_coords[2],
                                 score_coords[1] - score_coords[0],
                                 linewidth=1, edgecolor='r',
                                 facecolor='none', alpha=0.5)
        axs[0].add_patch(rect)

        rect = patches.Rectangle((gameover_coords[2], gameover_coords[0]),
                                 gameover_coords[3] - gameover_coords[2],
                                 gameover_coords[1] - gameover_coords[0],
                                 linewidth=1, edgecolor='r',
                                 facecolor='none', alpha=0.5)
        axs[0].add_patch(rect)

        f = axs[1].imshow(dino_frame, cmap="Greys_r")
        pl.colorbar(f, ax=axs[1])
        axs[2].imshow(score_frame, cmap="Greys_r")
        pl.show()

    print("d =", distance, "->", q_table[distance])

    if plotting:
        sys.exit(0)

np.save("q_table", q_table)
