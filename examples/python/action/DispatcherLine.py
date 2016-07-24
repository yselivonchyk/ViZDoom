import WallAlign as wah
import Discovery as dsc
import action as ac
import Initial as ini
import MapTracker as mp
import RunInLine
import InitialDelayHandler

# delay = 121
delay = 22

class DispatcherLine:
    """Class that react to the game"""
    currentHandler = None
    map = mp.MapTracker()

    def __init__(self):
        self.currentHandler = InitialDelayHandler.Delay()
        # self.currentHandler = ini.Initial()
        # self.currentHandler = dsc.Discovery()

    def handle(self, dmap, a):
        self.currentHandler.handle(dmap, a)
        if self.currentHandler.finished:
            print('!!!new handler')
            self.map = mp.MapTracker()
            self.currentHandler = RunInLine.RunInLine(self.map)

    def action(self):
        action =  self.currentHandler.action()
        self.map.register_action(action)
        return action