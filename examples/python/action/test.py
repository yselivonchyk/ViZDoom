import MapTracker as mp
import action as ac
import numpy as np

"""test map tracker path search"""

map = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 1, 2, 2, 1, 0],
    [0, 1, 2, 2, 1, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0]
])

print mp.MapTracker.candidate_quality(map, 0, 2, 3, (2, 4)) # 21
print mp.MapTracker.candidate_quality(map, 0, 2, 3, (3, 4)) # 20
print mp.MapTracker.candidate_quality(map, 0, 2, 3, (1, 3)) # 20
assert mp.MapTracker.candidate_quality(map, 0, 2, 3, (2, 4)) > mp.MapTracker.candidate_quality(map, 0, 2, 3, (1, 3))
print 270
print mp.MapTracker.candidate_quality(map, 270, 2, 3, (2, 4)) # 20
print mp.MapTracker.candidate_quality(map, 270, 2, 3, (3, 4)) # 20
print mp.MapTracker.candidate_quality(map, 270, 2, 3, (1, 3)) # 21

assert mp.MapTracker.candidate_quality(map, 270, 2, 3, (1, 3)) >  mp.MapTracker.candidate_quality(map, 270, 2, 3, (2, 4))
assert mp.MapTracker.candidate_quality(map, 90, 3, 2, (4, 2)) >  mp.MapTracker.candidate_quality(map, 270, 3, 2, (3, 1))
assert mp.MapTracker.candidate_quality(map, 180, 3, 2, (3, 1)) >  mp.MapTracker.candidate_quality(map, 270, 3, 2, (4, 2))