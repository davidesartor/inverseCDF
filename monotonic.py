from typing import Literal, Optional, Sequence, NamedTuple, Callable
import torch
import torch.nn as nn
from torch.nn.utils import parametrize
from torch.nn import functional as F

class Mapping(NamedTuple):
    forward: Callable[[torch.Tensor], torch.Tensor]
    right_inverse: Callable[[torch.Tensor], torch.Tensor]


class Positive(nn.Module):
    """Parametrization for positive parameters."""
    
    mappings = {
        'abs': Mapping(lambda x: x.abs(), lambda x: x),
        'exp': Mapping(lambda x: x.exp(), lambda x: x.abs().log()),
        'square': Mapping(lambda x: x.square(), lambda x: x.abs().sqrt()),
        'softplus': Mapping(lambda x: F.softplus(x), lambda x: x.expm1().log()),
        'celu': Mapping(lambda x: 1+torch.celu(x), lambda x: torch.where(x>1, x-1, x.log()))
    }

    def __init__(self, mapping:str):
        super().__init__()
        assert mapping in self.mappings.keys()
        self.mapping = mapping

    def forward(self, x):
        return self.mappings[self.mapping].forward(x)

    def right_inverse(self, x):
        return self.mappings[self.mapping].right_inverse(x)


class MonotonicLinear(nn.Linear):
    def __init__(
        self,
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        device=None, 
        dtype=None,
        weight_parametrization:Optional[str]=None,
        pre_activation=nn.Identity(),
    ):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.weight_parametrization = weight_parametrization
        self.act = pre_activation
        if self.weight_parametrization is not None:
            assert weight_parametrization in Positive.mappings.keys()
            parametrize.register_parametrization(
                self, 'weight', Positive(weight_parametrization)
            )
        
    def forward(self, x):
        x_pos = F.linear(self.act(x), self.weight.clamp(min=0.0), self.bias)
        if self.weight_parametrization is not None:
            return x_pos
        x_neg = F.linear(self.act(-x), self.weight.clamp(max=0.0))
        return x_pos + x_neg


class SwitchConvexity(nn.Module):
    def __init__(self, activation:nn.Module, dim:int, learnable=False):
        super().__init__()
        self.activation = activation
        if learnable:
            self.a = nn.Parameter(torch.rand(dim))
        else:
            self.register_buffer('a', (torch.rand(dim) > 0.5).float())
        
    def forward(self, x):
        a = self.a.clamp(0.0, 1.0).detach() + self.a - self.a.detach()
        return a*self.activation(x) + (1-a)*(-self.activation(-x))