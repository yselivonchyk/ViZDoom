import action as ac
import time as t
import Observer as ob
import Walker as wk
import MapTracker as mp

class Discovery:
    """Discover the map"""
    Observer, Walker = range(2)

    finished = False
    state = Observer

    map = None
    current = None

    def __init__(self, map):
        self.map = map
        self.current = ob.Observer(map)

    def action(self):
        return self.current.action()

    def handle(self, dmap, a):
        res = self.current.handle(dmap, a)
        if self.current.finished:
            if self.map.map_opened:
                t.sleep(1)
            if self.state == self.Observer:
                self.state = self.Walker
                self.current = wk.Walker(self.map)
            else:
                self.state = self.Observer
                self.current = ob.Observer(self.map)
                # self.current.handle(dmap, a)
        return res

