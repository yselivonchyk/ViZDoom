import action as ac
import numpy as np
import math
import time as t

class RunInLine:
    finished = False

    state = 0  # 0 - turn left
                # 1 - write left distance, turn around
                # 2 - write right distance, turn left
                # 3 - move along the line
    width = None
    position = None
    direction = True

    def __init__(self, map):
        print 'init:', self.__class__
        self.map = map

    def handle(self, dmap, a):
        if self.delay_loop > 0:
            return

        if self.state == 0:
            pass
        elif self.state == 1:
            dist = int(ac.central_dist(dmap))
            self.width = dist
            print('DIST:', dist)
        elif self.state == 2:
            print('second d',  ac.central_dist(dmap))
            self.width += ac.central_dist(dmap)
            self.position = ac.central_dist(dmap)
        else:
            self.direction = self.direction if self.position != self.width and self.position != 0 else not self.direction
        if self.state < 3:
            self.state += 1

    delay_action = None
    delay_loop = 0
    def delay(self):
        delay = 0
        self.delay_loop = self.delay_loop + 1 if self.delay_loop < delay else 0
        self.delay_action = ac.empty()
        return self.delay_loop > 0

    def action(self):
        if self.delay():
            return self.delay_action

        if self.state == 0:
            return ac.turn_left(ac.empty(), arg=90)
        elif self.state == 1:
            return ac.turn_left(ac.empty(), arg=180)
        elif self.state == 2:
            return ac.turn_left(ac.empty(), arg=90)
        else:
            self.position += 1 if self.direction else -1
            return ac.step_aside(ac.empty(), right=self.direction)
