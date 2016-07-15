import action as ac
import numpy as np
import math
import time as t

class WallAlign:
    """Aligns the player to the grid"""
    step = 4
    bestAngle = 0
    bestQuality = 0
    currentAngle = 0
    targetAngle = 360
    direction = True

    precise_mode = False

    finished = False

    move = ac.empty()
    current = None

    assigned = 0

    map = None

    def __init__(self, map):
        self.map = map

    def handle(self, dmap, a):
        angle = ac.turn_angle(a, self.move)
        q = WallAlign.quality(dmap)
        if q > self.bestQuality:
            self.bestQuality = q
            self.bestAngle = self.currentAngle

        self.print_state(q)
        self.currentAngle += angle

    _to_skip = 15

    def skipped(self):
        if self._to_skip > 0:
            self._to_skip -= 1
            return False
        return True

    dance_around = 0
    def action(self):
        self.move = ac.empty()
        self._turn_total += 1     # count the frames
        if not self.skipped(): return self.move

        dist = float(self.targetAngle - self.currentAngle)
        if math.fabs(dist/self.step) > 0.5:
            turn = int(self.step * np.sign(dist))
            self.assigned += turn
            self.move = ac.turn_left(self.move, turn)
        else:
            # assign new target
            if self.precise_mode:
                print('current best: %d %f' % (self.bestAngle, self.bestQuality))
                if self.dance_around == 0:
                    self.finished = True
                    self.map.reset_yaw()
                self.dance_around *= -1
                self.targetAngle = self.bestAngle
                self.move = ac.turn_left(ac.empty(), self.bestAngle - self.currentAngle + self.dance_around)
            else:
                self.move = self.assign_precise_search()
        return self.move

    def print_state(self, q = None):
        res = 'cur: %d \t targ: %d \t best: %d \t bq: %f' % (self.currentAngle, self.targetAngle, self.bestAngle, self.bestQuality)
        if q is not None:
            res += '\t Q: %f' % q
        print(res)

    def assign_precise_search(self):
        print('current best: %d %f' % (self.bestAngle, self.bestQuality))
        width = 1   # check in *width current steps apart
        move = ac.turn_left(ac.empty(), self.bestAngle + width * self.step - self.currentAngle)
        self.targetAngle = self.bestAngle - width * self.step
        self.step = 1
        self.precise_mode = True
        return move

    _turn_count = 0
    _turn_cycle = 1
    _turn_total = 0
    _turn_turns = 0

    def _turn_action(self):
        t = 5
        if self._turn_count == 0:
            self._turn_turns += 1
            print ('frame: %d \t turn: %d \t angle: %d' % (self._turn_total, self._turn_turns, self.currentAngle))
            self._turn_count = self._turn_cycle - 1
            return ac.turn_left(ac.empty(), t)
        else:
            self._turn_count -= 1
            return ac.empty()

    @staticmethod
    def dist(x, y):
        return np.abs(int(x) - y)

    @staticmethod
    def quality(depth):
        delta = 0
        y, x = ac.size(depth)
        y, x = y / 2, x / 2
        q = 0
        i, j, ref = x-1, x, depth[y, x]
        while i > 0:
            if WallAlign.dist(depth[y, i], ref) <= delta:
                depth[y, i] = 0
                depth[y + 1, i] = 229
                depth[y - 1, i] = 229
                q += 1
            else:
                break
            i -= 1
        while j < x * 2:
            if WallAlign.dist(depth[y, j], ref) <= delta:
                depth[y, j] = 229
                depth[y + 1, j] = 0
                depth[y - 1, j] = 249
                q += 1
            else:
                break
            j += 1
        return q

    @staticmethod
    def quality2(depth):
        y, x = ac.size(depth)
        y, x = y / 2, x / 2

        return q

