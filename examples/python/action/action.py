import numpy as np
import math
from time import sleep

__state = 'align'
__avoid = False

def empty():
    return [0] * 20


def size(depth):
    return len(depth), len(depth[0])


def turn_left(action, arg=20):
    action[18] += arg
    return action


def step(action, forward=True):
    if forward:
        action[6] = 1
    else:
        action[5] = 1
    return action

def step_aside(action, right=True):
    if right:
        action[3] = 1
    else:
        action[4] = 1
    return action


def check_region(depth, x, y, span, max_h, max_delta):
    height = 0
    i, j = y, y
    while i >= 0:
        if depth[i, x] == depth[y, x]:
            height += 1
            i -= 1
        else:
            break
    while j < max_h:
        if depth[j, x] == depth[y, x]:
            height += 1
            j += 1
        else:
            break
    return height > max_delta * 10


def action_available(dmap):
    height, width = len(dmap), len(dmap[0])
    return dmap[height / 3 * 2, width / 2]


def central_dist(dmap):
    height, width = len(dmap), len(dmap[0])
    return dmap[height / 2, width / 2]

def path_clear(dmap):
    height, width = len(dmap), len(dmap[0])
    min_dist = min(dmap[height / 2, width / 2], dmap[0, 0], dmap[-1, 0], dmap[-1, -1], dmap[0, -1])
    print 'mind', min_dist
    return min_dist > 0

def column_min_dist(dmap, x):
    return min(dmap[:, x])


def dist(x, y):
    return np.abs(int(x) - y)


def turn_angle(performed_action, assigned_action):
    angle = performed_action[18] + np.sign(assigned_action[18])
    # assert np.abs(performed_action[18]) < np.abs(angle) or angle == 0
    return angle

def turn_normalization(angle):
    if angle > 180:
        angle -= 360