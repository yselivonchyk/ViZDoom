import action as ac
import numpy as np
import math
import time as t

STEP = 1
TOTAL = 360*3

class RunInCircle:
    finished = False
    turn = 0

    def __init__(self, map):
        print 'init:', self.__class__
        self.map = map

    min_dist = 700
    min_angle = 0
    directions_checked = 0
    direction_selected = False

    def handle(self, dmap, a):
        if not self.direction_selected:
            dist = ac.central_dist(dmap)
            if dist < self.min_dist:
                self.min_dist = dist
                self.min_angle = self.directions_checked
        else:
            self.finished = self.turn >= TOTAL
            if self.turn >= TOTAL:
                exit(0)

    substep = 0
    def subturn(self):
        '''Turn by 1 degree and make a step every STEP turns'''
        self.turn += 1
        self.substep = self.substep + 1 if self.substep < STEP else 0
        action = ac.turn_left(ac.empty(), arg=1)

        # skip a bit once a circle
        if self.turn != 0 and self.turn % 360 == 0 and self.substep == 0:
            action = ac.empty()
            print('SKIPPED A BIT')

        if self.substep == 0:
            action = ac.step(action, forward=True)
        return action

    delay_action = None
    delay_loop = 0

    def action(self):
        if not self.direction_selected:
            self.directions_checked += 1
            if self.directions_checked == self.min_angle + 4:
                self.direction_selected = True
            return ac.turn_left(ac.empty(), arg=90)
        else:
            return self.subturn()


