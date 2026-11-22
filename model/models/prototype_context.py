"""Frozen distribution embedding with a compact, label-free prototype extension.

Preserve the base code instead of overwriting it with an ill-conditioned online
projection. Prototypes and bandwidth are fitted on base TRAIN codes only.
"""
import torch
from torch import Tensor, nn
from .transport_context import TransportContextEncoder

class PrototypeContextEncoder(nn.Module):

    def __init__(self, base_config=None, prototype_count=8, bandwidth_scale=1.0, clamp=12.0, robust=True):
        super().__init__()
        config = dict(base_config or {})
        architecture = config.pop('architecture', 'transport')
        if architecture != 'transport':
            raise ValueError('Prototype extension requires a transport encoder')
        if prototype_count < 1 or bandwidth_scale <= 0 or clamp <= 0:
            raise ValueError('Prototype count, bandwidth and clamp must be positive')
        self.encoder = TransportContextEncoder(**config)
        self.prototype_count = int(prototype_count)
        self.bandwidth_scale = float(bandwidth_scale)
        self.clamp = float(clamp)
        self.robust = bool(robust)
        dim = self.encoder.descriptor_dim
        self.descriptor_dim = dim + self.prototype_count
        self.register_buffer('code_mean', torch.zeros(dim))
        self.register_buffer('code_std', torch.ones(dim))
        self.register_buffer('centers', torch.zeros(self.prototype_count, dim))
        self.register_buffer('bandwidth', torch.ones(()))
        self.register_buffer('fitted', torch.tensor(False))

    @torch.no_grad()
    def fit_codes(self, codes: 'Tensor', seed=42):
        """Consumes codes only; labels / context IDs cannot enter this fit."""
        if bool(self.fitted):
            raise ValueError('Cannot refit after router statistics have been collected')
        if codes.ndim != 2 or codes.shape[1] != self.encoder.descriptor_dim or len(codes) < self.prototype_count:
            raise ValueError('Need >= prototype_count base training codes of the expected dimension')
        if not torch.isfinite(codes).all():
            raise ValueError('Codes must be finite')
        from sklearn.cluster import KMeans
        values = codes.detach().float().cpu()
        mean = values.mean(0)
        std = values.std(0, unbiased=False).clamp_min(0.001)
        normalized = ((values - mean) / std).clamp(-self.clamp, self.clamp)
        centers = torch.from_numpy(KMeans(n_clusters=self.prototype_count, n_init=10, random_state=seed).fit(normalized.numpy()).cluster_centers_)
        bandwidth = torch.cdist(normalized, centers).amin(1).median().clamp_min(0.1)
        self.code_mean.copy_(mean.to(self.code_mean))
        self.code_std.copy_(std.to(self.code_std))
        self.centers.copy_(centers.to(self.centers))
        self.bandwidth.copy_(bandwidth.to(self.bandwidth))
        self.fitted.fill_(True)
        self.eval().requires_grad_(False)

    def encode_codes(self, codes: 'Tensor') -> 'Tensor':
        if not bool(self.fitted):
            raise ValueError('Fit prototypes on base train before inference')
        q = ((codes - self.code_mean) / self.code_std).clamp(-self.clamp, self.clamp)
        retained = q.sign() * torch.log1p(q.abs()) if self.robust else q
        responses = torch.exp(-torch.cdist(q, self.centers).square() / (2 * (self.bandwidth_scale * self.bandwidth).square()))
        return torch.cat([retained, responses], dim=-1)

    def forward(self, points: 'Tensor') -> 'Tensor':
        return self.encode_codes(self.encoder(points))

    def get_config(self):
        return dict(architecture='transport_prototype', base_config=self.encoder.get_config(), prototype_count=self.prototype_count, bandwidth_scale=self.bandwidth_scale, clamp=self.clamp, robust=self.robust)
