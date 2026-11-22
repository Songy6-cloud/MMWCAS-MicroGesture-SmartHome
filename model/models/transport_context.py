"""Learned space-time distribution embedding, without point MLPs or convolutions.

Inspired by Pooling by Sliced-Wasserstein Embedding (NeurIPS 2021).
Finite quantiles + a learned compression approximate, not exactly preserve, SW.
"""
import torch
from torch import Tensor, nn
import torch.nn.functional as F

def projected_quantiles(cloud: 'Tensor', directions: 'Tensor', quantiles: 'int') -> 'Tensor':
    """Differentiable equal-mass sampling of each sorted 1-D projection."""
    projected = cloud @ F.normalize(directions, dim=-1).T
    ordered = projected.sort(dim=1).values
    positions = (torch.arange(quantiles, device=cloud.device, dtype=cloud.dtype) + 0.5) / quantiles
    positions = positions * (cloud.shape[1] - 1)
    lower, upper = (positions.floor().long(), positions.ceil().long())
    alpha = (positions - lower)[None, :, None]
    return (ordered[:, lower] * (1 - alpha) + ordered[:, upper] * alpha).flatten(1)

class TransportContextEncoder(nn.Module):

    def __init__(self, descriptor_dim: 'int'=64, projections: 'int'=16, quantiles: 'int'=16, frames: 'int'=28):
        super().__init__()
        if min(descriptor_dim, projections, quantiles, frames) < 1:
            raise ValueError('Encoder dimensions must be positive')
        self.descriptor_dim, self.projections = (descriptor_dim, projections)
        self.quantiles, self.frames = (quantiles, frames)
        self.register_buffer('input_mean', torch.zeros(3))
        self.register_buffer('input_std', torch.ones(3))
        self.directions = nn.Parameter(F.normalize(torch.randn(projections, 4), dim=-1))
        self.readout = nn.Linear(projections * quantiles, descriptor_dim)
        nn.init.orthogonal_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def spacetime(self, points: 'Tensor') -> 'Tensor':
        if points.ndim != 4 or points.shape[-1] < 3 or min(points.shape[1:3]) < 1:
            raise ValueError('points must have shape (B,T,N,C), with nonempty T,N and C>=3')
        xyz = (points[..., :3] - self.input_mean) / self.input_std
        time = torch.linspace(-1, 1, points.shape[1], device=points.device, dtype=points.dtype)
        time = time[None, :, None, None].expand(*points.shape[:3], 1)
        return torch.cat([xyz, time], dim=-1).flatten(1, 2)

    def forward(self, points: 'Tensor') -> 'Tensor':
        return self.readout(projected_quantiles(self.spacetime(points), self.directions, self.quantiles))

    def get_config(self) -> 'dict':
        return dict(architecture='transport', descriptor_dim=self.descriptor_dim, projections=self.projections, quantiles=self.quantiles, frames=self.frames)

def distribution_geometry_loss(code: 'Tensor', reference: 'Tensor') -> 'Tensor':
    """Preserve label-free relative distances between empirical distributions."""
    learned = torch.pdist(code) / code.shape[-1] ** 0.5
    target = torch.pdist(reference.detach()) / reference.shape[-1] ** 0.5
    learned = learned / learned.mean().clamp_min(1e-06)
    target = target / target.mean().clamp_min(1e-06)
    return F.smooth_l1_loss(learned, target)
