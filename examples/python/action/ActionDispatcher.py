import action as ac

class ActionDispatcher:
    """tests all available actions"""
    count = 0
    frequency = 24

    def action(self):
        self.count += 1

        a = ac.empty()
        i = (self.count / self.frequency) % len(a)

        a[i] = 1
        if self.count % self.frequency == 0:
            print(self.count, i, a)
        return a

    def handle(self, dmap, a):
        pass