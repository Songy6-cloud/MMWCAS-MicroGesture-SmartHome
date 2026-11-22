"""Fixed TensorSketch interaction for semantic and unsupervised context codes.

Compact bilinear pooling (Gao et al., CVPR 2016) supplies the sketch mechanism.
Here the map stays fixed after base fitting so analytic CL statistics stay valid.
This module is a router feature map, not an unsupervised context encoder.
"""
import torch
from torch import Tensor
from .context_router import QuadraticTrajectoryRouterTransform

class ContextInteractionTransform(QuadraticTrajectoryRouterTransform):

    def __init__(self, clamp=12.0, input_dim=0, code_dim=72, sketch_dim=256, sketch_seed=1729):
        super().__init__(clamp=clamp, input_dim=input_dim)
        if min(code_dim, sketch_dim) < 1 or (input_dim and input_dim <= code_dim):
            raise ValueError('Interaction needs positive dimensions and both semantic/context branches')
        self.code_dim = int(code_dim)
        self.sketch_dim = int(sketch_dim)
        self.sketch_seed = int(sketch_seed)
        for name in ('semantic_hash', 'context_hash'):
            self.register_buffer(name, torch.empty(0, dtype=torch.long))
        for name in ('semantic_sign', 'context_sign'):
            self.register_buffer(name, torch.empty(0))
        if input_dim:
            self._initialize_sketch(input_dim)

    def _initialize_sketch(self, input_dim):
        semantic_dim = input_dim - self.code_dim
        if semantic_dim < 1:
            raise ValueError('Missing semantic branch')
        rng = torch.Generator().manual_seed(self.sketch_seed)
        self.semantic_hash = torch.randint(self.sketch_dim, (semantic_dim,), generator=rng)
        self.context_hash = torch.randint(self.sketch_dim, (self.code_dim,), generator=rng)
        self.semantic_sign = torch.randint(2, (semantic_dim,), generator=rng).float() * 2 - 1
        self.context_sign = torch.randint(2, (self.code_dim,), generator=rng).float() * 2 - 1

    @torch.no_grad()
    def fit_base(self, router_inputs):
        super().fit_base(router_inputs)
        self._initialize_sketch(router_inputs.shape[1])

    def sketch(self, semantic: 'Tensor', context: 'Tensor') -> 'Tensor':
        if semantic.ndim != 2 or context.ndim != 2 or semantic.shape[0] != context.shape[0]:
            raise ValueError('Need aligned 2-D semantic/context matrices')
        if semantic.shape[1] != len(self.semantic_hash) or context.shape[1] != len(self.context_hash):
            raise ValueError('Feature dimensions do not match the fixed sketch')
        batch = len(semantic)
        left = semantic.new_zeros(batch, self.sketch_dim).scatter_add_(1, self.semantic_hash.to(semantic.device)[None].expand(batch, -1), semantic * self.semantic_sign.to(semantic))
        right = context.new_zeros(batch, self.sketch_dim).scatter_add_(1, self.context_hash.to(context.device)[None].expand(batch, -1), context * self.context_sign.to(context))
        return torch.fft.irfft(torch.fft.rfft(left) * torch.fft.rfft(right), n=self.sketch_dim)

    def forward(self, router_inputs):
        basic = super().forward(router_inputs)
        n = router_inputs.shape[1]
        semantic = n - self.code_dim
        interaction = self.sketch(basic[:, :semantic], basic[:, semantic:n])
        return torch.cat([basic, interaction], dim=-1)

    def get_config(self):
        return dict(architecture='context_interaction', clamp=self.clamp, input_dim=self.base_mean.numel(), code_dim=self.code_dim, sketch_dim=self.sketch_dim, sketch_seed=self.sketch_seed)
