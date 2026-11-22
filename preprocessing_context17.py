"""Raw finite XYZ -> deterministic local interpolation to 120 XYZ points.

Python 3.6 compatible. No torch or PU-Net dependency in this module.
Matches the 2026-11-13 offline 13-capture interpolation experiment.
"""
import numpy as np
PIPELINE = 'no_filter_raw_xyz_interp120_v1'


def fps_unique(xyz, count=120):
    xyz = np.asarray(xyz, dtype=np.float32)
    if len(xyz) <= count:
        return xyz.copy()
    chosen = []; used = np.zeros(len(xyz), bool)
    distance = np.full(len(xyz), np.inf); index = 0
    for _ in range(count):
        chosen.append(index); used[index] = True
        distance = np.minimum(distance, np.sum((xyz-xyz[index])**2, axis=1))
        distance[used] = -1
        index = int(distance.argmax())
    return xyz[chosen]


def interpolate120(xyz):
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('Expected finite N x 3 XYZ')
    if len(xyz) >= 120:
        return fps_unique(xyz, 120)
    if len(xyz) == 1:
        return np.repeat(xyz, 120, axis=0)
    if len(xyz) == 0:
        raise ValueError('Empty frame requires explicit temporal handling')
    delta = xyz[:, None, :] - xyz[None, :, :]
    distance = np.sum(delta**2, axis=2)
    np.fill_diagonal(distance, np.inf)
    neighbors = np.argsort(distance, axis=1, kind='mergesort')[:, :min(3, len(xyz)-1)]
    edges = sorted({tuple(sorted((i, int(j)))) for i in range(len(xyz)) for j in neighbors[i]})
    missing = 120-len(xyz)
    allocations = np.full(len(edges), missing//len(edges), int)
    allocations[:missing % len(edges)] += 1
    added = []
    for (i, j), count in zip(edges, allocations):
        for step in range(1, count+1):
            alpha = step/(count+1.0)
            added.append((1-alpha)*xyz[i]+alpha*xyz[j])
    return np.concatenate([xyz, np.array(added, np.float32).reshape(-1, 3)])


class InterpolationPreprocessor:
    def __init__(self, zero_radius_m=.05):
        if not np.isfinite(zero_radius_m) or zero_radius_m < 0:
            raise ValueError('Invalid origin removal radius')
        self.zero_radius_m = float(zero_radius_m)

    def run(self, clouds):
        if not len(clouds):
            raise ValueError('No frames to preprocess')
        filtered, details = [], []
        for cloud in clouds:
            xyz = np.asarray(cloud, dtype=np.float64)[:, :3]
            if xyz.shape[1] != 3 or not np.isfinite(xyz).all():
                raise ValueError('Invalid XYZ cloud')
            # Context 17 was trained and tested on the raw board XYZ points.
            # Keep every finite point; only the fixed 120-point shape adapter
            # below is applied before EF-CIR inference.
            clean = xyz
            info = dict(input_count=len(xyz), filtered_count=len(xyz),
                        removed_origin=0, deduplicated_count=len(xyz),
                        isolated_removed=0, filtering_disabled=True)
            filtered.append(clean); details.append(info)
        valid = np.array([i for i, frame in enumerate(filtered) if len(frame)], dtype=int)
        sources = [i if len(frame) else (int(valid[np.argmin(abs(valid-i))]) if len(valid) else -1)
                   for i, frame in enumerate(filtered)]
        targets = np.linspace(0, len(clouds)-1, 28)
        indices = abs(np.arange(len(clouds))[:, None]-targets[None, :]).argmin(axis=0)
        points = np.zeros((28, 120, 3), np.float32)
        cache = {}
        for target, observed in enumerate(indices):
            source = sources[int(observed)]
            if source < 0:
                continue  # Entirely empty capture is rejected by Recognizer.
            if source not in cache:
                cache[source] = interpolate120(filtered[source])
            points[target] = cache[source]
        for i, info in enumerate(details):
            info['source_frame_after_empty_fill'] = sources[i]+1 if sources[i] >= 0 else None
            info['imputed'] = sources[i] != i
            info['effective_ratio'] = 120.0/len(filtered[sources[i]]) if sources[i] >= 0 else None
        return points, None, {
            'pipeline': PIPELINE, 'frames': details,
            'frame_indices_zero_based': indices.tolist(),
            'source_frame_indices_zero_based': [sources[int(i)] for i in indices],
            'empty_frames_1based': [i+1 for i, frame in enumerate(filtered) if not len(frame)],
            'parameters': {'point_filtering': False, 'origin_removal': False},
            'sampling120': 'preserve raw finite XYZ points; deterministic nearest-neighbor edge interpolation; FPS above 120; repeat singleton',
            'empty_frames': 'copy nearest nonempty raw frame, tie earlier; original valid-frame counts retained; reject all-empty capture',
            'coordinates': 'unchanged MMWCAS XYZ, meters', 'punet_used': False}
