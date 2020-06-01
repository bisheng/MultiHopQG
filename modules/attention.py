# -*- coding: utf-8 -*-

""" 
Created at 2019-06-13 10:30:53
==============================
attention 
"""

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class Attention(nn.Module):

    def __init__(self, dim):
        super(Attention, self).__init__()

        self.temperature = np.power(dim, 0.5)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=2)
        self.layer_norm = nn.LayerNorm(dim)
        self.p_gen = nn.Linear(dim, 1)

    def forward(self, output, context, mask=None):
        residual = output
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        attn = attn / self.temperature
        if mask is not None:
            attn.data.masked_fill_(mask, -float('inf'))

        out_attn=F.log_softmax(attn,dim=-1)
        attn = self.softmax(attn)
        attn = self.dropout(attn)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        output = torch.bmm(attn, context)
        # output=self.dropout(output)

        # output = self.layer_norm(output + residual)

        # p_gen = self.p_gen(output + residual).squeeze(2)

        return output, out_attn #, p_gen
