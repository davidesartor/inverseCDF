from typing import Literal, Sequence
import torch
import torch.nn as nn
from torch.nn import functional as F

class MonotonicBlock(nn.Module):
    def __init__(self, in_dim:int, out_dim:int, pre_activation=nn.Identity()):
        super().__init__()
        xavier_scale = (6/ (in_dim + out_dim)) ** 0.5
        self.W = nn.Parameter(xavier_scale*torch.rand((out_dim, in_dim)))
        self.b = nn.Parameter(torch.zeros(out_dim))
        self.act = pre_activation

    def forward(self, x):
        W_pos, W_neg = self.W.clamp(min=0), self.W.clamp(max=0)
        x = self.act(x) @ W_pos.T + self.act(-x) @ W_neg.T + self.b
        return x

class MonotonicMLP(nn.Sequential):
    def __init__(
            self, 
            dims: Sequence[int], 
            activation:nn.Module, 
            input_activation=nn.Identity(),
            output_activation=nn.Identity(),
        ):
        layers = [MonotonicBlock(dims[0], dims[1]), input_activation]
        for in_dim, out_dim in zip(dims[1:-1], dims[2:], ):
            layers.append(MonotonicBlock(in_dim, out_dim, activation))
        layers.append(output_activation)
        super().__init__(*layers)


class ConstrainedLinear(nn.Module):
    def __init__(self, in_dim:int, out_dim:int, weight_activation:str):
        super().__init__()
        self.w_act, self.w_act_inv = {
            'abs': (torch.abs, torch.abs),
            'square': (torch.square, torch.sqrt),
            'exp': (torch.exp, torch.log),
            'softplus': (F.softplus, lambda x: x.expm1().log()),
            'identity': (lambda x: x, lambda x: x)
        }[weight_activation.lower()]

        xavier_scale = (6/ (in_dim + out_dim)) ** 0.5
        W = xavier_scale*torch.rand((out_dim, in_dim))
        self.W = nn.Parameter(self.w_act_inv(W))
        self.b = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        return F.linear(x, self.w_act(self.W), self.b)


class RandomConvexityBreak(nn.Module):
    def __init__(self, activation, dim):
        super().__init__()
        self.act = activation
        self.a = torch.rand(dim) > 0.5
        
    def forward(self, x):
        return torch.where(self.a, self.act(x), -self.act(-x))


class LearnableConvexityBreak(nn.Module):
    def __init__(self, activation, dim):
        super().__init__()
        self.act = activation
        self.a = nn.Parameter(torch.randn(dim))

    def forward(self, x):
        a = torch.sigmoid(self.a)
        return a*self.act(x) + (1-a)*(-self.act(-x))


class ConstrainedMLP(nn.Sequential):
    def __init__(
            self, 
            dims: Sequence[int], 
            activation:nn.Module, 
            output_activation=nn.Identity(),
            weight_activation:Literal['abs', 'square', 'exp', 'identity']='abs',
            convexity_break:Literal['random', 'learnable', 'none']='random'
        ):
        assert convexity_break in ['random', 'learnable', 'none']
        make_activation = {
            'random': lambda dim: RandomConvexityBreak(activation, dim),
            'learnable': lambda dim: LearnableConvexityBreak(activation, dim),
            'none': lambda dim: activation
        }[convexity_break]

        layers = [] 
        for din, dout in zip(dims[:-1], dims[1:]):
            layers.append(ConstrainedLinear(din, dout, weight_activation))
            layers.append(make_activation(dout))
        layers[-1] = output_activation
        super().__init__(*layers)
        

