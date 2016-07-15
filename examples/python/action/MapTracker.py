import time as t
import math
import numpy as np
import Queue as queue
import action as ac

class MapTracker:
    Unknown, Seen, Visited, Blocked = range(4)

    map_opened = False

    size = 512
    x, y = size/2, size/2
    yaw, pitch = 0, 0
    # map = np.negative(np.ones((size, size), np.uint8))
    map = np.zeros((size, size), np.uint8)

    path = None
    target_x, target_y = 0, 0

    def __init__(self):
        self.update_current(MapTracker.Visited)

    def move(self, v, w):
        self.x += v
        self.y += w
        return self.update_current(MapTracker.Visited)

    def move_view(self, offset):
        print 'move view offset', offset
        delta = self._decompensate_yaw(offset)
        print 'move view delta', delta, self.yaw
        self.x += delta[0]
        self.y += delta[1]
        return self.update_current(MapTracker.Visited)

    def see(self, v, w, blocked=False):
        """set state Seen to tile that has (v, w) offset from current position"""
        new_value = MapTracker.Blocked if blocked else MapTracker.Seen
        res = self.update_offset(v, w, new_value)
        return res

    def see_offset(self, offset, blocked=False):
        offset = self._compensate_yaw(offset)
        new_value = MapTracker.Blocked if blocked else MapTracker.Seen
        res = self.update_offset(offset[0], offset[1], new_value)
        return res

    def update_offset(self, v, w, arg):
        curr = self.map[self.x + v, self.y + w]
        if arg < curr:
            print 'no luck (new, old): (%d, %d)' % (arg, curr)
            return
        self.map[self.x + v, self.y + w] = arg
        return curr

    def update_current(self, arg):
        return self.update_offset(0, 0, arg)

    def current(self, new=-1):
        return self.map[self.x, self.y]

    def get_offset(self, v, w):
        return self.map[self.x + v, self.y + w]

    def get_view(self, offset):
        offset = self._compensate_yaw(offset)
        return self.get_offset(offset[0], offset[1])

    def move_to(self, next_tile):
        curr = (self.x, self.y)
        delta = (next_tile[0] - curr[0], next_tile[1] - curr[1])
        print 'delta', delta
        assert delta[0] < 2
        assert delta[1] < 2
        self.move(delta[0], delta[1])
        action = self.move_to_action(ac.empty(), delta)
        print 'act', action
        return action

    def move_to_action(self, action, delta):
        if delta[1] != 0:
            action = ac.step(action, forward=delta[1] > 0)
        if delta[0] != 0:
            action = ac.step_aside(action, right=delta[0] > 0)
        return action

    def move_to_faceforward(self, next_tile):
        curr = (self.x, self.y)
        delta = (-curr[0] +next_tile[0], -curr[1] +next_tile[1])
        assert (delta[0] < 2 and delta[1] == 0) or (delta[1] < 2 and delta[0] == 0)
        offset = self._decompensate_yaw(delta)
        print 'move offset to delta %s -%3d-> %s' % (delta, self.yaw, offset)
        if offset[0] == 0:
            if offset[1] < 0:
                return ac.turn_left(ac.empty(), arg = 180), next_tile
            self.x, self.y = next_tile
            return self.move_to_action(ac.empty(), offset), None
        else:
            ang = 90 if offset[0] > 1 else 270
            return ac.turn_left(ac.empty(), arg = ang), next_tile

    path_map= None
    closest_quality = None
    closest_dist = None
    closest_point = None

    def path_to_unvisited(self):
        self.find_closest_unvisited()
        print 'closest point', self.closest_point

        if self.closest_point is None:
            self.print_current(5)

        if self.closest_point is None:
            self.map_opened = True
            return

        self.path = self.build_path_to_point((self.x, self.y), self.closest_point, self.path_map)
        return self.path

    def print_current(self, dist=7):
        # print 'current locatin (%d, %d) a=%d: \n\r' % (self.x, self.y, self.yaw), np.swapaxes(np.array(self.map[self.x - dist:self.x + dist + 1, self.y - dist:self.y + dist+ 1]), 0, 1)
        print 'current locatin (%d, %d) a=%d: \n\r' % (self.x, self.y, self.yaw), np.array(self.map[self.x - dist:self.x + dist + 1, self.y - dist:self.y + dist + 1])

    @staticmethod
    def build_path_to_point(start_point, stop_point, path_map):
        curr_dist = path_map[stop_point]
        path = []
        while stop_point != start_point:
            path.append(stop_point)
            curr_dist -= 1
            # search next hop
            ngb = MapTracker.neighbours(stop_point)
            for x in ngb:
                if path_map[x] == curr_dist:
                    stop_point = x
                    break
        path.reverse()
        return path

    def find_closest_unvisited(self):
        min_weight = 1

        self.closest_dist, self.closest_point, self.closest_quality = None, None, 0
        self.path_map = np.zeros(self.map.shape, np.uint8)
        self.path_map[self.x, self.y] = min_weight
        q = queue.PriorityQueue()

        q.put((min_weight, (self.x, self.y)))
        while not q.empty():
            last_val, last = q.get(0)
            if self.closest_dist is not None and last_val >= self.closest_dist:
                continue

            # add neighbours when they deserve it
            for candidate in self.neighbours(last):
                if self.queue_check_add(last_val, candidate, q):
                    last = candidate
                    continue

    def queue_check_add(self, last_val, candidate, q):
        w = last_val + 1
        if self.map[candidate] == MapTracker.Seen:
            current_quality = self._candidate_quality(candidate)
            if self.closest_point is None or current_quality > self.closest_quality:
                print 'new winner', candidate, current_quality
                self.path_map[candidate] = w
                self.closest_quality = current_quality
                self.closest_dist = w
                self.closest_point = candidate
            return True
        if self.map[candidate] == MapTracker.Visited and (self.path_map[candidate] == 0 or self.path_map[candidate] > w):
            q.put((w, candidate))
            self.path_map[candidate] = w
            return False

    def register_action(self, action):
        # print 'a18', action[18]
        self.yaw += action[18]
        if self.yaw > 360:
            self.yaw -= 360
        if self.yaw < 0:
            self.yaw += 360

    def reset_yaw(self):
        self.yaw = 0

    @staticmethod
    def neighbours(tile):
        return [(tile[0] + 1, tile[1]), (tile[0], tile[1] + 1), (tile[0]-1, tile[1]),   (tile[0], tile[1] - 1)]

    @staticmethod
    def neighbours_all(tile):
        return [(tile[0] + 1, tile[1]), (tile[0], tile[1] + 1), (tile[0]-1, tile[1]),   (tile[0], tile[1] - 1),
                (tile[0] + 1, tile[1]+1), (tile[0] + 1, tile[1]-1), (tile[0]-1, tile[1]+1), (tile[0]-1, tile[1]-1)]

    def _compensate_yaw(self, offset):
        return MapTracker.compensate_yaw(offset, self.yaw)

    def _decompensate_yaw(self, offset):
        return MapTracker.compensate_yaw(offset, 360 - self.yaw % 360)

    @staticmethod
    def compensate_yaw(offset, yaw):
        # print 'yaw', yaw
        if yaw % 90 != 0:
            print yaw
        assert yaw % 90 == 0

        if yaw == 90:
            return (offset[1], -offset[0])
        if yaw == 180:
            return (-offset[0], -offset[1])
        if yaw == 270:
            return (-offset[1], offset[0])
        return offset

        # if yaw % 180 == 90:
        #     offset = offset[1], -1 * offset[0]
        # if yaw >= 180:
        #     offset = -1 * offset[0], -1 * offset[1]
        # return offset

    def _candidate_quality(self, candidate):
        return self.candidate_quality(self.map, self.yaw, self.x, self.y, candidate)

    @staticmethod
    def normalize_yaw(yaw):
        v = yaw / 360
        if v != 0:
            print 'norm', yaw, '->', yaw - 360*v
        yaw -= 360*v
        return yaw

    @staticmethod
    def candidate_quality(map, yaw, x, y, candidate):
        yaw = MapTracker.normalize_yaw(yaw)
        assert yaw < 360 and yaw >= 0
        quality = 0
        # prefer points close to already observed points
        for n in MapTracker.neighbours_all(candidate):
            if map[n] == MapTracker.Visited:
                quality += 1
        # prefer points that does not require to turn
        quality *= 10
        offset = -x + candidate[0], -y + candidate[1]
        delta = MapTracker.compensate_yaw(offset, 360-yaw)
        if delta[0] == 0 and delta[1] > 0:
            quality += 1
            print 'better q', offset, '-%d->' % yaw, delta

        return quality