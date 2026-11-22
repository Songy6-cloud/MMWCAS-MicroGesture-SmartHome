"""Python 3.6 / PyTorch 1.10 compatibility and bounded inference batches."""
import inspect
import os
from pathlib import Path
import torch
MICRO_BATCH = max(1, int(os.environ.get('NANO_MICRO_BATCH', '2')))
EVAL_BATCH = max(1, int(os.environ.get('NANO_EVAL_BATCH', '4')))
DATA_ROOT = os.environ.get('LPTF_DATA_ROOT', '')

def load_checkpoint(path, map_location='cpu', weights_only=True, mmap=False):
    options = dict(map_location=map_location)
    params = inspect.signature(torch.load).parameters
    if 'weights_only' in params:
        options['weights_only'] = weights_only
    return torch.load(path, **options)

def data_root(manifest):
    return Path(DATA_ROOT) if DATA_ROOT else Path(manifest).resolve().parent

def forward_chunks(model, points):
    return torch.cat([model(x) for x in points.split(EVAL_BATCH)], dim=0)

def backward_batch(model, points, labels):
    """One optimizer batch, micro-batched activations; caller clips/steps once."""
    total_loss = 0.0
    correct = 0
    for x, y in zip(points.split(MICRO_BATCH), labels.split(MICRO_BATCH)):
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y, reduction='sum') / len(labels)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Non-finite training loss')
        loss.backward()
        total_loss += float(loss.detach())
        correct += int((logits.detach().argmax(-1) == y).sum())
    return (total_loss, correct)

class _StableGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.nn.functional.gelu(x)

    @staticmethod
    def backward(ctx, grad):
        import math
        x, = ctx.saved_tensors
        derivative = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        derivative = derivative + x * torch.exp(-0.5 * x.square()) / math.sqrt(2.0 * math.pi)
        return grad * derivative

class NanoGELU(torch.nn.Module):
    """Same exact GELU forward; stable analytical derivative for PyTorch1.10."""
    def forward(self, x):
        return _StableGELU.apply(x)
