import action as ac
import numpy as np
import math
import time as t
import collector

LENGTH = 8  # use even nubmer
SIDE_REPEAT = 3
WAIT_NO_CHANGE = 2

PHASES = [
    (True, True),
    (False, True),
    (False, False),
    (True, False)
]


def set_8_trajectory():
    global PHASES
    PHASES = [
        (True, True),
        (False, True),
        (False, True),
        (True, True),
        (True, False),
        (False, False),
        (False, False),
        (True, False)
    ]


class RunInRomb:
    finished = False
    turn = 0

    def __init__(self, map):
        print 'init:', self.__class__
        print(PHASES)
        self.map = map

    step = 0

    def handle(self, dmap, a):
        self.finished = self.step > len(PHASES)*LENGTH
        if self.step > len(PHASES)*LENGTH:
            print('survival rate %d/%d' % (collector._actual_count, collector._counter))
            exit(0)

    similar_frames = 0
    side_step_action = None
    side_step_repeat = 0

    def action(self):
        self.similar_frames = self.similar_frames + 1 if not collector.image_changed else 0

        time_for_action = self.similar_frames >= WAIT_NO_CHANGE
        if not time_for_action:
            return ac.empty()

        if self.side_step_repeat > 0:
            self.side_step_repeat -= 1
            return self.side_step_action
        self.similar_frames = 0
        action = ac.empty()

        phase = PHASES[int(self.step / LENGTH)%len(PHASES)]
        substep = self.step % LENGTH
        # print(phase, substep, substep % 2)
        use_x_axis = substep % 2 == 0
        # make steps along extremes longer
        if self.step / LENGTH % 2 == 1:
            use_x_axis = not use_x_axis
        print('ph:%d ss:%d/%d F:%s  \t%d \t%s' % (self.step / LENGTH,
                                                  substep + 1,
                                                  LENGTH,
                                                  str(use_x_axis),
                                                  collector._counter,
                                                  str(phase)))
        if use_x_axis:
            action = ac.step(action, forward=phase[0])
        else:
            self.side_step_action = ac.step_aside(action, right=phase[1])
            self.side_step_repeat = SIDE_REPEAT - 1
            action = self.side_step_action
        collector.register_action(action)
        self.step += 1
        return action


