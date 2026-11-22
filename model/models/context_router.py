import torch
from torch import Tensor, nn

class QuadraticTrajectoryRouterTransform(nn.Module):
    """Fixed base-normalized quadratic map for an analytic context router."""

    def __init__(self, clamp: 'float'=12.0, input_dim: 'int'=0):
        super().__init__()
        self.clamp = float(clamp)
        self.register_buffer('base_mean', torch.zeros(input_dim))
        self.register_buffer('base_std', torch.ones(input_dim))

    @property
    def is_fitted(self) -> 'bool':
        return self.base_mean.numel() > 0

    @torch.no_grad()
    def fit_base(self, router_inputs: 'Tensor') -> 'None':
        if self.is_fitted:
            raise ValueError('The trajectory router transform is already fitted')
        if router_inputs.ndim != 2 or router_inputs.shape[0] == 0:
            raise ValueError('router_inputs must have shape (samples, features)')
        values = router_inputs.detach().float().cpu()
        self.base_mean = values.mean(dim=0)
        self.base_std = values.std(dim=0, unbiased=False).clamp_min(0.001)

    def forward(self, router_inputs: 'Tensor') -> 'Tensor':
        if not self.is_fitted:
            raise ValueError('Fit the transform on the base context first')
        mean = self.base_mean.to(router_inputs.device, router_inputs.dtype)
        std = self.base_std.to(router_inputs.device, router_inputs.dtype)
        normalized = ((router_inputs - mean) / std).clamp(-self.clamp, self.clamp)
        return torch.cat([normalized, normalized.square()], dim=-1)

    def get_config(self) -> 'dict[str, float | int]':
        return {'input_dim': int(self.base_mean.numel()), 'output_dim': int(self.base_mean.numel() * 2), 'clamp': self.clamp}

def context_router_input(points, semantic_features, encoder):
    return torch.cat([semantic_features, encoder(points)], dim=-1)
