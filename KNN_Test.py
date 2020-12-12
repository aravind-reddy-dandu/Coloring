import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import numpy as np
from itertools import count


def make_kd_tree(points, dim, i=0):
    if len(points) > 1:
        points.sort(key=lambda x: x[i])
        i = (i + 1) % dim
        half = len(points) >> 1
        return [
            make_kd_tree(points[: half], dim, i),
            make_kd_tree(points[half + 1:], dim, i),
            points[half]
        ]
    elif len(points) == 1:
        return [None, None, points[0]]


def get_knn(kd_node, point, k, dim, dist_func, return_distances=True, i=0, heap=None):
    import heapq
    is_root = not heap
    if is_root:
        heap = []
    if kd_node is not None:
        dist = dist_func(point, kd_node[2])
        dx = kd_node[2][i] - point[i]
        if len(heap) < k:
            heapq.heappush(heap, (-dist, next(tiebreaker), kd_node[2]))
        elif dist < -heap[0][0]:
            heapq.heappushpop(heap, (-dist, next(tiebreaker), kd_node[2]))
        i = (i + 1) % dim
        # Goes into the left branch, and then the right branch if needed
        for b in [dx < 0] + [dx >= 0] * (dx * dx < -heap[0][0]):
            get_knn(kd_node[b], point, k, dim, dist_func, return_distances, i, heap)
    if is_root:
        neighbors = sorted((-h[0], h[1]) for h in heap)
        return neighbors if return_distances else [n[1] for n in neighbors]


dim = 9


def dist_sq(a, b, dim):
    return sum((a[i] - b[i]) ** 2 for i in range(dim))


def dist_sq_dim(a, b):
    return dist_sq(a, b, dim)


tiebreaker = count()
patches = np.array(pd.read_csv('D:\\Study\\Intro_AI\\Coloring_Assignment\\patches/patches.csv'))
right_patches = np.array(pd.read_csv('D:\\Study\\Intro_AI\\Coloring_Assignment\\patches/right_patches.csv'))
sc_tree = make_kd_tree(list(patches), 9)
neighs = []
for pt in tqdm(right_patches[:1]):
    neighs.append(get_knn(sc_tree, list(pt), 6, dim, dist_sq_dim, return_distances=False))
print(neighs)
nbrs = NearestNeighbors(n_neighbors=6).fit(patches)
distances, nearest_neighbours = nbrs.kneighbors(right_patches[:1], 6, return_distance=True)
print(nearest_neighbours)



