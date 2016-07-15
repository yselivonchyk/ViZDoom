import action as ac


class Initial:
    finished = False

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
        self.finished = self.skipped()
        return ac.empty()
