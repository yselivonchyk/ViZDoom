import action as ac
import numpy as np
import math
import time as t

class Delay:
    """Wait initial delay"""
    finished = False

    def __init__(self):
        pass

    def handle(self, dmap, a):
        pass

    _to_skip = 15

    def skipped(self):
        if self._to_skip > 0:
            self._to_skip -= 1
            return False
        return True

    dance_around = 0
    def action(self):
        if self.skipped():
            self.finished = True
        return ac.empty()
