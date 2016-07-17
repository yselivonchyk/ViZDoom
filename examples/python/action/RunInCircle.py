import action as ac
import numpy as np
import math
import time as t

class RunInCircle:
    STEP = 2
    TOTAL = 360

    finished = False
    turn = 0

    def __init__(self, map):
        print 'init:', self.__class__
        self.map = map

    min_dist = 700
    min_angle = 0
    angle = 0
    direction_selected = False

    def handle(self, dmap, a):
        if not self.direction_selected:
            dist = ac.central_dist(dmap)
            if dist < self.min_dist:
                self.min_dist = dist
                self.min_angle = self.angle
        else:
            self.finished = self.turn >= self.TOTAL
            if self.turn >= self.TOTAL:
                exit(0)

    def action(self):
        if not self.direction_selected:
            self.angle += 1
            if self.angle == self.min_angle + 4:
                self.direction_selected = True
            return ac.turn_left(ac.empty(), arg=90)
        else:
            self.turn += self.STEP
            return ac.turn_left(ac.step(ac.empty(), forward=True), arg=self.STEP)


