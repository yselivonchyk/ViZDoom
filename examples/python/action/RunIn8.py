import action as ac
import numpy as np
import math
import time as t

class RunIn8:
    STEP = 5
    TOTAL = 360

    finished = False
    turn = 0
    turn_back = 0

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


    def action(self):
        if not self.direction_selected:
            self.angle += 1
            if self.angle == self.min_angle + 4:
                self.direction_selected = True
            return ac.turn_left(ac.empty(), arg=90)

        if self.turn % 360 == 0:
            self.turn_back = 0;
        if self.turn == self.TOTAL + 2*self.STEP:
            exit(0)

        else:
            if (self.turn % 360) != 180 or self.turn_back == 360:
                self.turn += self.STEP
                return ac.turn_left(ac.step(ac.empty(), forward=True), arg=self.STEP)
            else:
                self.turn_back += self.STEP
                return ac.turn_left(ac.step(ac.empty(), forward=True), arg=-self.STEP)


