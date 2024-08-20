from typing import Literal, Optional, Sequence, NamedTuple, Callable
import torch
import torch.nn as nn
from torch.nn.utils import parametrize
from torch.nn import functional as F

class MonotonicLinear(nn.Linear):
    def __init__(
        self,
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        device=None, 
        dtype=None,
        pre_activation=nn.Identity(),
    ):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.act = pre_activation
        
    def forward(self, x):
        w_pos = self.weight.clamp(min=0.0)
        w_neg = self.weight.clamp(max=0.0)
        x_pos = F.linear(self.act(x), w_pos, self.bias)
        x_neg = F.linear(self.act(-x), w_neg, self.bias)  
        return x_pos + x_neg

class MonotonicLinearPost(nn.Linear):
    def __init__(
        self,
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        device=None, 
        dtype=None,
        post_activation=nn.Identity(),
    ):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.act = post_activation
        
    def forward(self, x):
        w_pos = self.weight.clamp(min=0.0)
        w_neg = self.weight.clamp(max=0.0)
        x_pos = self.act(F.linear(x, w_pos, self.bias))
        x_neg = -self.act(F.linear(x, w_neg, self.bias))
        return x_pos + x_neg


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
        'cholesky': Mapping(lambda x: torch.cholesky(x), lambda x: x @ x.t())
    }

    def __init__(self, mapping:str):
        super().__init__()
        assert mapping in self.mappings.keys()
        self.mapping = mapping

    def forward(self, x):
        return self.mappings[self.mapping].forward(x)

    def right_inverse(self, x):
        return self.mappings[self.mapping].right_inverse(x)


class ConstrainedLinear(nn.Linear):
    def __init__(
        self,
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        device=None, 
        dtype=None,
        weight_parametrization:'abs',
    ):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        assert weight_parametrization in Positive.mappings.keys()
        parametrize.register_parametrization(
            self, 'weight', Positive(weight_parametrization)
        )

class SwitchConvexity(nn.Module):
    def __init__(self, activation:nn.Module, dim:int, learnable=False):
        super().__init__()
        self.activation = activation
        if learnable:
            self.a = nn.Parameter(torch.rand(dim))
        else:
            self.register_buffer('a', (torch.rand(dim) > 0.5).float())
        
    def forward(self, x):
        if isinstance(self.a, torch.nn.Parameter):
            self.a = torch.nn.Parameter(self.a.clamp(0.0, 1.0))
        return self.a*self.activation(x) + (1-self.a)*(-self.activation(-x))