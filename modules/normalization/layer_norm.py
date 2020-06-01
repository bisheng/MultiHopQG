# -*- coding: utf-8 -*-

""" 
Created at 2019-06-10 16:15:41
==============================
normalization 
"""
import torch
from torch import nn, optim
import torch.functional as F

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct b layernorm module in the TF style (epsilon inside the square root).
        等于nn.LayerNorm(input.size(-1))
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias