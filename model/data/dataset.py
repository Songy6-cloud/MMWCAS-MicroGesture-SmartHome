import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
TEXT_SUFFIXES = {'.txt', '.xyz'}

def load_context_config(path: 'str | Path') -> 'dict[str, Any]':
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

@dataclass(frozen=True)
class ManifestRow:
    path: 'Path'
    label: 'int'
    context: 'str'
    context_id: 'int'
    split: 'str'
    metadata: 'dict[str, str]'

def _load_text_points(path: 'Path', frames: 'int', points_per_frame: 'int') -> 'np.ndarray':
    """Load ordered txt/xyz point clouds.

    Expected format: one xyz point per line, comma- or whitespace-separated.
    The file order is interpreted as 28 frames x 120 points by default:
    rows 0..119 -> frame 0, rows 120..239 -> frame 1, and so on.
    """
    try:
        arr = np.loadtxt(path, delimiter=',', dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
    except ValueError:
        arr = np.loadtxt(path, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
    if arr.shape[1] < 3:
        raise ValueError('Expected at least 3 columns x,y,z in {0}, got shape {1}'.format(path, arr.shape))
    arr = arr[:, :3].astype('float32', copy=False)
    expected = frames * points_per_frame
    if arr.shape[0] != expected:
        raise ValueError('{0} has {1} points, but ordered txt/xyz format requires exactly {2} points for {3} frames x {4} points.'.format(path, arr.shape[0], expected, frames, points_per_frame))
    if not np.isfinite(arr).all():
        raise ValueError('{0} contains NaN or infinite coordinates'.format(path))
    return arr.reshape(frames, points_per_frame, 3)

def _load_points(path: 'Path', frames: 'int', points_per_frame: 'int') -> 'np.ndarray':
    suffix = path.suffix.lower()
    if suffix == '.npy':
        arr = np.load(path)
    elif suffix == '.npz':
        archive = np.load(path)
        if 'points' not in archive:
            raise KeyError("{0} does not contain a 'points' array".format(path))
        arr = archive['points']
    elif suffix in TEXT_SUFFIXES:
        return _load_text_points(path, frames=frames, points_per_frame=points_per_frame)
    else:
        raise ValueError('Unsupported sample format: {0}'.format(path.suffix))
    if arr.ndim != 3:
        raise ValueError('Expected sample shape (T, N, C), got {0} from {1}'.format(arr.shape, path))
    return arr.astype('float32', copy=False)

def _fit_shape(points: 'np.ndarray', frames: 'int', points_per_frame: 'int') -> 'np.ndarray':
    t, n, c = points.shape
    out = np.zeros((frames, points_per_frame, c), dtype=np.float32)
    use_t = min(t, frames)
    use_n = min(n, points_per_frame)
    out[:use_t, :use_n] = points[:use_t, :use_n]
    if use_t < frames and use_t > 0:
        out[use_t:] = out[use_t - 1]
    return out

def load_point_cloud_sample(path: 'str | Path', frames: 'int'=28, points_per_frame: 'int'=120) -> 'Tensor':
    """Load one inference sample as a ``(frames, points, channels)`` tensor."""
    sample_path = Path(path)
    points = _load_points(sample_path, frames, points_per_frame)
    points = _fit_shape(points, frames, points_per_frame)
    return torch.from_numpy(points)

class ManifestPointCloudDataset(Dataset):
    """Manifest-driven dataset for mmWave point-cloud sequences."""

    def __init__(self, manifest_path: 'str | Path', context_names: 'list[str]', split: 'str'='train', frames: 'int'=28, points_per_frame: 'int'=120, root: 'str | Path | None'=None, num_classes: 'int | None'=None):
        self.manifest_path = Path(manifest_path)
        self.root = Path(root) if root is not None else self.manifest_path.parent
        self.frames = frames
        self.points_per_frame = points_per_frame
        self.context_to_id = {name: i for i, name in enumerate(context_names)}
        rows: 'list[ManifestRow]' = []
        with open(self.manifest_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for raw in reader:
                context = raw['context']
                if context not in self.context_to_id or raw.get('split', 'train') != split:
                    continue
                sample_path = Path(raw['path'])
                if not sample_path.is_absolute():
                    sample_path = self.root / sample_path
                label = int(raw['label'])
                if num_classes is not None and (not 0 <= label < num_classes):
                    raise ValueError('Label {0} in {1} is outside [0, {2}] for path={3}'.format(label, self.manifest_path, num_classes - 1, raw['path']))
                metadata = {k: v for k, v in raw.items() if k not in {'path', 'label', 'context', 'split'}}
                rows.append(ManifestRow(path=sample_path, label=label, context=context, context_id=self.context_to_id[context], split=raw.get('split', split), metadata=metadata))
        if not rows:
            raise ValueError('No rows found for split={0} and contexts={1}'.format(split, context_names))
        self.rows = rows

    def __len__(self) -> 'int':
        return len(self.rows)

    def __getitem__(self, index: 'int') -> 'dict[str, Tensor | str]':
        row = self.rows[index]
        points = load_point_cloud_sample(row.path, self.frames, self.points_per_frame)
        return {'points': points, 'label': torch.tensor(row.label, dtype=torch.long), 'context_id': torch.tensor(row.context_id, dtype=torch.long), 'context': row.context, 'path': str(row.path)}

def context_names_from_config(config: 'dict[str, Any]', include_multi_person: 'bool'=False) -> 'list[str]':
    names = [item['name'] for item in config['known_contexts']]
    if include_multi_person:
        names.extend((item['name'] for item in config.get('multi_person_contexts', [])))
    return names
