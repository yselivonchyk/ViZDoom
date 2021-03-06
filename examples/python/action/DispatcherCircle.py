import WallAlign as wah
import Discovery as dsc
import action as ac
import Initial as ini
import MapTracker as mp
import RunIn8
import RunInCircle
import RunInRomb
import collector

delay = 121


RUN_TRAJECTORY = None


class DispatcherCircle:
    """Class that react to the game"""
    currentHandler = None
    map = mp.MapTracker()

    def __init__(self):
        self.currentHandler = wah.WallAlign(self.map)
        # self.currentHandler = ini.Initial()
        # self.currentHandler = dsc.Discovery()

    def handle(self, dmap, a):
        self.currentHandler.handle(dmap, a)
        if self.currentHandler.finished:
            print('!!!new handler')
            self.map = mp.MapTracker()
            collector.start_recording()
            if RUN_TRAJECTORY is None:
                self.currentHandler = RunIn8.RunInCircle(map)
            else:
                self.currentHandler = RUN_TRAJECTORY(map)
            # self.currentHandler = RunInCircle.RunInCircle(map)
            self.currentHandler = RunInRomb.RunInRomb(map)

    def action(self):
        action = self.currentHandler.action()
        self.map.register_action(action)
        return action