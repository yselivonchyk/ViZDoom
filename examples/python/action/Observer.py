import action as ac
import MapTracker as mp
class Observer:
    finished = False
    map = None
    move = ac.empty()

    def __init__(self, map):
        print 'Observer'
        self.map = map

    def action(self):
        return self.move

    def handle(self, dmap, a):
        min_d = 2
        center_blocked = ac.central_dist(dmap) == min_d
        next_left = ac.column_min_dist(dmap, 0) > min_d
        next_right = ac.column_min_dist(dmap, -1) > min_d

        # save visual data
        self.map.see_offset((0, 1), blocked=center_blocked)
        if next_left:
            self.map.see_offset((-1, 1))
        if next_right:
            self.map.see_offset((1, 1))
        self.map.print_current()
        print self.map.get_view((-1, 1)), self.map.get_view((1, 1)), next_left, next_right, self.map.yaw

        # turn to the next unobserved point
        angle = 0
        current_offset = (0, 1)
        for i in range(4):
            current_offset = mp.MapTracker.compensate_yaw(current_offset, 90)
            angle += 90
            if self.map.get_view(current_offset) == mp.MapTracker.Unknown:
                self.move = ac.turn_left(ac.empty(), arg=angle)
                return
        self.finished = True
        self.move = ac.empty()