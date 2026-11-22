from nano_runtime import NanoGELU
from dataclasses import dataclass
import torch
from torch import Tensor, nn

def _std(values: 'Tensor', mean: 'Tensor', dim: 'int') -> 'Tensor':
    variance = (values - mean.unsqueeze(dim)).square().mean(dim=dim)
    return variance.clamp_min(1e-08).sqrt()

class PointShapeFrameEncoder(nn.Module):
    """Encode local point-cloud shape without graph construction over points."""

    def __init__(self, in_channels: 'int'=3, width: 'int'=64, token_dim: 'int'=96):
        super().__init__()
        if in_channels < 3:
            raise ValueError('The first three channels must be xyz')
        point_input_dim = 4 + in_channels - 3
        self.point_mlp = nn.Sequential(nn.Linear(point_input_dim, width), nn.LayerNorm(width), NanoGELU(), nn.Linear(width, width), NanoGELU(), nn.Linear(width, width))
        self.point_attention = nn.Linear(width, 1)
        self.output = nn.Sequential(nn.Linear(width * 4, token_dim), nn.LayerNorm(token_dim), NanoGELU())

    def forward(self, points: 'Tensor') -> 'Tensor':
        batch, frames, num_points, channels = points.shape
        xyz = points[..., :3]
        center = xyz.mean(dim=2)
        spread = _std(xyz, center, dim=2).clamp_min(0.001)
        normalized = (xyz - center.unsqueeze(2)) / spread.unsqueeze(2)
        radius = normalized.norm(dim=-1, keepdim=True)
        point_features = torch.cat([normalized, radius, points[..., 3:]], dim=-1)
        hidden = self.point_mlp(point_features.reshape(batch * frames, num_points, channels + 1))
        mean_pool = hidden.mean(dim=1)
        std_pool = _std(hidden, mean_pool, dim=1)
        max_pool = hidden.amax(dim=1)
        attention = torch.softmax(self.point_attention(hidden).squeeze(-1), dim=1)
        attention_pool = (hidden * attention.unsqueeze(-1)).sum(dim=1)
        token = self.output(torch.cat([mean_pool, std_pool, max_pool, attention_pool], dim=-1))
        return token.reshape(batch, frames, -1)

class TrajectoryNodeEncoder(nn.Module):
    """Turn per-frame point-cloud centers into graph node embeddings."""

    def __init__(self, token_dim: 'int'=96, hidden_dim: 'int'=48):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(13, hidden_dim), nn.LayerNorm(hidden_dim), NanoGELU(), nn.Linear(hidden_dim, token_dim), nn.LayerNorm(token_dim), NanoGELU())

    def forward(self, points: 'Tensor') -> 'Tensor':
        xyz = points[..., :3]
        center = xyz.mean(dim=2)
        spread = xyz.std(dim=2, unbiased=False)
        velocity = torch.zeros_like(center)
        velocity[:, 1:] = center[:, 1:] - center[:, :-1]
        acceleration = torch.zeros_like(center)
        acceleration[:, 2:] = velocity[:, 2:] - velocity[:, 1:-1]
        speed = velocity.norm(dim=-1, keepdim=True)
        nodes = torch.cat([center, spread, velocity, acceleration, speed], dim=-1)
        return self.encoder(nodes)

class PointTrajectoryFrameEncoder(nn.Module):
    """Shared dual branch: local point shape and absolute center trajectory."""

    def __init__(self, in_channels: 'int'=3, point_width: 'int'=64, token_dim: 'int'=96, trajectory_width: 'int'=48):
        super().__init__()
        self.token_dim = token_dim
        self.output_dim = token_dim * 2
        self.point_encoder = PointShapeFrameEncoder(in_channels, point_width, token_dim)
        self.trajectory_encoder = TrajectoryNodeEncoder(token_dim, trajectory_width)

    def forward(self, points: 'Tensor') -> 'Tensor':
        if points.ndim != 4:
            raise ValueError('points must have shape (B,T,N,C)')
        point_tokens = self.point_encoder(points)
        trajectory_nodes = self.trajectory_encoder(points)
        return torch.cat([point_tokens, trajectory_nodes], dim=-1)

class TemporalGraphBlock(nn.Module):
    """Message passing on a fixed multi-scale temporal trajectory graph."""

    def __init__(self, channels: 'int'=96, bottleneck: 'int'=48, dropout: 'float'=0.1):
        super().__init__()
        self.message = nn.Sequential(nn.Linear(channels, bottleneck), NanoGELU(), nn.Dropout(dropout), nn.Linear(bottleneck, channels))
        self.gate = nn.Linear(channels * 2, 1)
        self.norm = nn.LayerNorm(channels)

    @staticmethod
    def adjacency(frames: 'int', device: 'torch.device', dtype: 'torch.dtype') -> 'Tensor':
        matrix = torch.eye(frames, device=device, dtype=dtype)
        for offset, weight in ((1, 1.0), (2, 0.5), (4, 0.25)):
            indices = torch.arange(frames - offset, device=device)
            matrix[indices, indices + offset] = weight
            matrix[indices + offset, indices] = weight
        return matrix / matrix.sum(dim=-1, keepdim=True)

    def forward(self, nodes: 'Tensor') -> 'Tensor':
        adjacency = self.adjacency(nodes.shape[1], nodes.device, nodes.dtype)
        neighborhood = torch.einsum('ts,bsd->btd', adjacency, nodes)
        message = self.message(neighborhood)
        gate = torch.sigmoid(self.gate(torch.cat([nodes, neighborhood], dim=-1)))
        return self.norm(nodes + gate * message)

class PointTemporalBlock(nn.Module):

    def __init__(self, channels: 'int'=96, dropout: 'float'=0.1):
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.pointwise = nn.Sequential(nn.Conv1d(channels, channels, kernel_size=1), NanoGELU(), nn.Dropout(dropout))
        self.norm = nn.LayerNorm(channels)

    def forward(self, tokens: 'Tensor') -> 'Tensor':
        update = self.pointwise(self.depthwise(tokens.transpose(1, 2))).transpose(1, 2)
        return self.norm(tokens + update)

class GraphTrajectoryTemporalMixer(nn.Module):
    """Graph trajectory reasoning, point fusion, and learned frame attention."""

    def __init__(self, token_dim: 'int'=96, graph_depth: 'int'=3, graph_bottleneck: 'int'=48, dropout: 'float'=0.1):
        super().__init__()
        self.token_dim = token_dim
        self.point_temporal = PointTemporalBlock(token_dim, dropout)
        self.graph_blocks = nn.ModuleList([TemporalGraphBlock(token_dim, graph_bottleneck, dropout) for _ in range(graph_depth)])
        self.fusion = nn.Sequential(nn.Linear(token_dim * 2, token_dim), nn.LayerNorm(token_dim), NanoGELU())
        self.frame_attention = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, token_dim // 2), nn.Tanh(), nn.Linear(token_dim // 2, 1))

    def forward(self, tokens: 'Tensor', return_attention: 'bool'=False) -> 'tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]':
        point_tokens, trajectory_nodes = tokens.split(self.token_dim, dim=-1)
        point_tokens = self.point_temporal(point_tokens)
        for block in self.graph_blocks:
            trajectory_nodes = block(trajectory_nodes)
        fused = self.fusion(torch.cat([point_tokens, trajectory_nodes], dim=-1))
        attention = torch.softmax(self.frame_attention(fused).squeeze(-1), dim=-1)
        summary = (fused * attention.unsqueeze(-1)).sum(dim=1)
        if return_attention:
            return (summary, fused, attention)
        return (summary, fused)

@dataclass
class LPTFOutput:
    logits: 'Tensor'
    features: 'Tensor'
    frame_tokens: 'Tensor'
    frame_attention: 'Tensor'

class LPTFNet(nn.Module):
    """Lightweight point/trajectory graph network for micro-gesture recognition."""
    model_type = 'lptf_net'

    def __init__(self, in_channels: 'int'=3, num_classes: 'int'=12, num_contexts: 'int'=0, point_width: 'int'=64, token_dim: 'int'=96, trajectory_width: 'int'=48, graph_depth: 'int'=3, graph_bottleneck: 'int'=48, dropout: 'float'=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.num_contexts = num_contexts
        self.frame_encoder = PointTrajectoryFrameEncoder(in_channels, point_width, token_dim, trajectory_width)
        self.temporal = GraphTrajectoryTemporalMixer(token_dim, graph_depth, graph_bottleneck, dropout)
        self.head = nn.Sequential(nn.LayerNorm(token_dim), nn.Dropout(dropout), nn.Linear(token_dim, num_classes))

    def forward(self, points: 'Tensor', context_ids: 'Tensor | None'=None, return_features: 'bool'=False) -> 'Tensor | LPTFOutput':
        del context_ids
        tokens = self.frame_encoder(points)
        features, mixed_tokens, attention = self.temporal(tokens, return_attention=True)
        logits = self.head(features)
        if return_features:
            return LPTFOutput(logits=logits, features=features, frame_tokens=mixed_tokens, frame_attention=attention)
        return logits

    @torch.no_grad()
    def expand_classes(self, new_num_classes: 'int') -> 'None':
        if new_num_classes <= self.num_classes:
            return
        old = self.head[-1]
        new = nn.Linear(old.in_features, new_num_classes, device=old.weight.device)
        nn.init.kaiming_uniform_(new.weight, a=5 ** 0.5)
        nn.init.zeros_(new.bias)
        new.weight[:self.num_classes].copy_(old.weight)
        new.bias[:self.num_classes].copy_(old.bias)
        self.head[-1] = new
        self.num_classes = new_num_classes

    def get_config(self) -> 'dict[str, int | float]':
        point_encoder = self.frame_encoder.point_encoder
        trajectory_encoder = self.frame_encoder.trajectory_encoder
        first_graph = self.temporal.graph_blocks[0]
        return {'in_channels': point_encoder.point_mlp[0].in_features - 1, 'num_classes': self.num_classes, 'num_contexts': self.num_contexts, 'point_width': point_encoder.point_mlp[0].out_features, 'token_dim': self.temporal.token_dim, 'trajectory_width': trajectory_encoder.encoder[0].out_features, 'graph_depth': len(self.temporal.graph_blocks), 'graph_bottleneck': first_graph.message[0].out_features, 'dropout': float(self.head[1].p)}
