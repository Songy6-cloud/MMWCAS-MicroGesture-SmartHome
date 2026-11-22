import copy
import math
from dataclasses import dataclass
import torch
from torch import Tensor, nn
import torch.nn.functional as F

class TemporalContextExpert(nn.Module):
    """Context-isolated temporal mixer and local classifier."""

    def __init__(self, base_temporal: 'nn.Module', feature_dim: 'int', global_labels: 'list[int]', base_head: 'nn.Sequential | None'=None, dropout: 'float'=0.1, classifier_type: 'str'='linear'):
        super().__init__()
        if classifier_type != 'linear':
            raise ValueError('Unsupported expert classifier')
        self.classifier_type = classifier_type
        if not global_labels:
            raise ValueError('An expert requires at least one global label')
        self.temporal = copy.deepcopy(base_temporal)
        self.register_buffer('global_labels', torch.tensor(global_labels, dtype=torch.long))
        compatible = base_head is not None
        if compatible and len(global_labels) == base_head[-1].out_features:
            self.head = copy.deepcopy(base_head)
        else:
            self.head = nn.Sequential(nn.LayerNorm(feature_dim), nn.Dropout(dropout), nn.Linear(feature_dim, len(global_labels)))
            if base_head is not None:
                self.head[0].load_state_dict(base_head[0].state_dict())
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    def train(self, mode: 'bool'=True):
        super().train(mode)
        if not any((p.requires_grad for p in self.temporal.parameters())):
            self.temporal.eval()
        return self

    def forward(self, frame_tokens: 'Tensor') -> 'Tensor':
        features, _ = self.temporal(frame_tokens)
        return self.head(features)

    def global_log_probabilities(self, frame_tokens: 'Tensor', num_classes: 'int') -> 'Tensor':
        local_log_probabilities = F.log_softmax(self(frame_tokens), dim=-1)
        output = frame_tokens.new_full((frame_tokens.shape[0], num_classes), -torch.inf)
        output[:, self.global_labels] = local_log_probabilities
        return output

@dataclass
class ContextSufficientStats:
    """Second-order moments for one context; no sample features are retained."""
    count: 'int'
    feature_sum: 'Tensor'
    feature_outer_sum: 'Tensor'

    def state_dict(self) -> 'dict[str, object]':
        return {'count': self.count, 'feature_sum': self.feature_sum.cpu(), 'feature_outer_sum': self.feature_outer_sum.cpu()}

    @classmethod
    def from_state_dict(cls, state: 'dict[str, object]') -> "'ContextSufficientStats'":
        return cls(count=int(state['count']), feature_sum=state['feature_sum'].cpu(), feature_outer_sum=state['feature_outer_sum'].cpu())

class AnalyticContextRouter:
    """Class-balanced ridge router learned from replay-free sufficient statistics.

    Each context is consumed once. Only its count, feature sum, and feature
    outer-product sum are retained. Re-fitting after a new context therefore
    never needs historical point clouds or historical feature vectors.
    """

    def __init__(self, ridge: 'float'=0.1, temperature: 'float'=0.25):
        if ridge <= 0:
            raise ValueError('ridge must be positive')
        if temperature <= 0:
            raise ValueError('temperature must be positive')
        self.ridge = float(ridge)
        self.temperature = float(temperature)
        self.contexts: 'list[str]' = []
        self.statistics: 'dict[str, ContextSufficientStats]' = {}
        self.feature_mean: 'Tensor | None' = None
        self.feature_std: 'Tensor | None' = None
        self.classifier: 'Tensor | None' = None

    @torch.no_grad()
    def add_context(self, context: 'str', features: 'Tensor') -> 'None':
        if context in self.statistics:
            raise ValueError('Router already contains context={0}'.format(context))
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError('features must have shape (samples, feature_dim)')
        values = features.detach().float().cpu()
        self.contexts.append(context)
        self.statistics[context] = ContextSufficientStats(count=values.shape[0], feature_sum=values.sum(dim=0), feature_outer_sum=values.T @ values)
        self._fit()

    @torch.no_grad()
    def _fit(self) -> 'None':
        if not self.contexts:
            raise ValueError('Cannot fit a router without contexts')
        first = self.statistics[self.contexts[0]]
        feature_dim = first.feature_sum.numel()
        total_count = sum((self.statistics[name].count for name in self.contexts))
        total_sum = sum((self.statistics[name].feature_sum for name in self.contexts), torch.zeros(feature_dim, dtype=torch.float64))
        total_outer = sum((self.statistics[name].feature_outer_sum for name in self.contexts), torch.zeros(feature_dim, feature_dim, dtype=torch.float64))
        mean = total_sum / total_count
        variance = (total_outer.diagonal() / total_count - mean.square()).clamp_min(1e-08)
        std = variance.sqrt()
        design_dim = feature_dim + 1
        num_contexts = len(self.contexts)
        gram = torch.zeros(design_dim, design_dim, dtype=torch.float64)
        targets = torch.zeros(design_dim, num_contexts, dtype=torch.float64)
        context_mass = total_count / num_contexts
        std_outer = std[:, None] * std[None, :]
        mean_outer = mean[:, None] * mean[None, :]
        for index, context in enumerate(self.contexts):
            stats = self.statistics[context]
            context_mean = stats.feature_sum / stats.count
            context_second = stats.feature_outer_sum / stats.count
            centered_second = context_second - mean[:, None] * context_mean[None, :] - context_mean[:, None] * mean[None, :] + mean_outer
            standardized_mean = (context_mean - mean) / std
            standardized_second = centered_second / std_outer
            design_second = torch.zeros(design_dim, design_dim, dtype=torch.float64)
            design_second[:-1, :-1] = standardized_second
            design_second[:-1, -1] = standardized_mean
            design_second[-1, :-1] = standardized_mean
            design_second[-1, -1] = 1.0
            gram.add_(design_second, alpha=context_mass)
            targets[:-1, index] = context_mass * standardized_mean
            targets[-1, index] = context_mass
        identity = torch.eye(design_dim, dtype=torch.float64)
        classifier = torch.linalg.solve(gram + self.ridge * identity, targets)
        self.feature_mean = mean.float()
        self.feature_std = std.float()
        self.classifier = classifier.float()

    def scores(self, features: 'Tensor', contexts: 'list[str] | None'=None) -> 'Tensor':
        if self.classifier is None or self.feature_mean is None or self.feature_std is None:
            raise ValueError('The analytic router has not been fitted')
        selected = self.contexts if contexts is None else contexts
        unknown = [context for context in selected if context not in self.statistics]
        if unknown:
            raise ValueError('Unknown router contexts: {0}'.format(unknown))
        mean = self.feature_mean.to(device=features.device, dtype=features.dtype)
        std = self.feature_std.to(device=features.device, dtype=features.dtype)
        standardized = (features - mean) / std
        design = torch.cat([standardized, torch.ones_like(standardized[:, :1])], dim=-1)
        indices = [self.contexts.index(context) for context in selected]
        classifier = self.classifier[:, indices].to(device=features.device, dtype=features.dtype)
        return design @ classifier

    def weights(self, features: 'Tensor', contexts: 'list[str] | None'=None) -> 'Tensor':
        return torch.softmax(self.scores(features, contexts) / self.temperature, dim=-1)

    def state_dict(self) -> 'dict[str, object]':
        if self.classifier is None or self.feature_mean is None or self.feature_std is None:
            raise ValueError('The analytic router has not been fitted')
        return {'ridge': self.ridge, 'temperature': self.temperature, 'contexts': self.contexts, 'statistics': {context: self.statistics[context].state_dict() for context in self.contexts}, 'feature_mean': self.feature_mean.cpu(), 'feature_std': self.feature_std.cpu(), 'classifier': self.classifier.cpu()}

    @classmethod
    def from_state_dict(cls, state: 'dict[str, object]') -> "'AnalyticContextRouter'":
        router = cls(ridge=float(state['ridge']), temperature=float(state['temperature']))
        router.contexts = list(state['contexts'])
        router.statistics = {context: ContextSufficientStats.from_state_dict(state['statistics'][context]) for context in router.contexts}
        router.feature_mean = state['feature_mean'].cpu()
        router.feature_std = state['feature_std'].cpu()
        router.classifier = state['classifier'].cpu()
        return router

class EFCIRExpertBank(nn.Module):
    """Frozen backbone companion that isolates all context-specific updates."""

    def __init__(self, num_classes: 'int'):
        super().__init__()
        self.num_classes = int(num_classes)
        self.experts = nn.ModuleDict()

    def add_expert(self, context: 'str', expert: 'TemporalContextExpert') -> 'None':
        if context in self.experts:
            raise ValueError('Expert already exists for context={0}'.format(context))
        self.experts[context] = expert

    def oracle_logits(self, features: 'Tensor', context: 'str') -> 'Tensor':
        return self.experts[context].global_log_probabilities(features, self.num_classes)

    def mixture_logits(self, features: 'Tensor', router: 'AnalyticContextRouter', contexts: 'list[str] | None'=None, router_features: 'Tensor | None'=None) -> 'Tensor':
        selected = list(self.experts.keys()) if contexts is None else contexts
        routing_input = features if router_features is None else router_features
        router_log_weights = router.weights(routing_input, selected).clamp_min(1e-08).log()
        expert_outputs = torch.stack([self.experts[context].global_log_probabilities(features, self.num_classes) for context in selected], dim=1)
        return torch.logsumexp(expert_outputs + router_log_weights.unsqueeze(-1), dim=1)
