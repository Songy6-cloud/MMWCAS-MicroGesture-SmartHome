"""Python 3.6 port of variableIsolationForestPointCloudFilter.m (XYZ only)."""
import numpy as np


def unique_xyz(points):
    xyz = np.asarray(points, dtype=np.float64)[:, :3]
    if not np.isfinite(xyz).all():
        raise ValueError('Nonfinite XYZ')
    _, first = np.unique(xyz, axis=0, return_index=True)
    return xyz[np.sort(first)]


def normalizer(n):
    if n <= 1:
        return 0.0
    return 2.0 * np.sum(1.0 / np.arange(1, n)) - 2.0 * (n - 1) / n


def filter_xyz(points, height=8, trees=100, radius=.2, min_neighbors=8,
               threshold=.5, flat_tolerance=.05, seed=42):
    """Deduplicate first; count self; estimate k then take <=k scores >0.5.

    Matches existing MATLAB max(score)<=0.55 flat-score rule. Each frame
    resets the random seed. No sklearn dependency or contamination limit.
    """
    xyz = unique_xyz(points)
    n = len(xyz)
    rng = np.random.RandomState(seed)
    delta = xyz[:, None, :] - xyz[None, :, :]
    neighbors = np.sum(np.sum(delta * delta, axis=2) < radius ** 2, axis=1)
    estimated = int(np.sum(neighbors < min_neighbors))
    path_sum = np.zeros(n, dtype=np.float64)

    def grow(indices, depth):
        if len(indices) <= 1 or depth >= height:
            path_sum[indices] += depth + normalizer(len(indices))
            return
        subset = xyz[indices]
        lo, hi = subset.min(axis=0), subset.max(axis=0)
        dims = np.flatnonzero(hi > lo)
        if not len(dims):
            path_sum[indices] += depth + normalizer(len(indices))
            return
        dim = dims[int(rng.random_sample() * len(dims))]
        split = lo[dim] + rng.random_sample() * (hi[dim] - lo[dim])
        left = subset[:, dim] < split
        if left.all() or not left.any():
            path_sum[indices] += depth + normalizer(len(indices))
            return
        grow(indices[left], depth + 1)
        grow(indices[~left], depth + 1)

    if n > 1:
        for _ in range(trees):
            grow(np.arange(n), 0)
        scores = 2.0 ** (-path_sum / trees / normalizer(n))
    else:
        scores = np.full(n, .5)
    flat = not n or float(scores.max()) <= threshold + flat_tolerance
    removed = np.empty(0, dtype=int)
    if estimated and not flat:
        candidates = np.flatnonzero(scores > threshold)
        order = np.argsort(-scores[candidates], kind='mergesort')
        removed = np.sort(candidates[order[:estimated]])
    keep = np.ones(n, dtype=bool)
    keep[removed] = False
    info = {'raw_count': len(points), 'unique_count': n,
            'filtered_count': int(keep.sum()), 'estimated_outliers': estimated,
            'removed_indices': removed.tolist(), 'scores': scores.tolist(),
            'neighbor_counts': neighbors.tolist(), 'scores_flat': bool(flat)}
    return xyz[keep], info


def fit30(xyz):
    """FPS above 30; cyclic repeat below 30. Empty frames remain empty."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if not len(xyz):
        return np.zeros((30, 3), dtype=np.float32)
    if len(xyz) <= 30:
        return xyz[np.arange(30) % len(xyz)].astype(np.float32)
    chosen = [0]
    distance = np.full(len(xyz), np.inf)
    for _ in range(29):
        distance = np.minimum(distance, np.sum((xyz - xyz[chosen[-1]]) ** 2, axis=1))
        chosen.append(int(np.argmax(distance)))
    return xyz[chosen].astype(np.float32)
