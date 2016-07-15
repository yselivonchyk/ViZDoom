import action as ac
import MapTracker as mp

class Walker:
    state = 0
    finished = False
    map = None

    def __init__(self, map):
        self.map = map

    path = None
    target_tile = None
    def action(self):
        if self.path is None:
            path = self.map.path_to_unvisited()

            if path is None:
                print 'YOUVE SEEN EVERYTHING, BOY'
                return ac.empty()

            self.path = iter(path)

        if self.target_tile is None:
            self.target_tile = next(self.path, None)

        if self.target_tile is None:
            self.path = None
            self.finished = True
            return ac.empty()

        next_action, self.target_tile = self.map.move_to_faceforward(self.target_tile)
        return next_action
        # return self.map.move_to(self.target_tile)

    def handle(self, dmap, a):
        self.map.update_current(mp.MapTracker.Visited)
        pass