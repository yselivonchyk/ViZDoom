import WallAlign as wah
import Discovery as dsc
import action as ac
import Initial as ini
import MapTracker as mp
import RunIn8
import RunInCircle

delay = 121

class DummyDispatcher:
    def handle(self, dmap, a):
        pass

    def action(self):
        return ac.empty()