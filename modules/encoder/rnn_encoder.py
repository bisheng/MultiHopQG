# -*- coding: utf-8 -*-

""" 
Created at 2019-06-09 21:36:53
==============================
simple_encoder 
"""
import torch
from torch import nn, optim
import torch.functional as F


class RNNEncoder(nn.Module):

    def __init__(self, vocab_size, emb_dim=256, hidden_size=512, dropout=0.1, padding_idx=None, bidirectional=False,
                 embedding=None):
        super(RNNEncoder, self).__init__()

        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        # if embedding is not None:
        #     self.embedding.weight = nn.Parameter(embedding)
        # else:
        #     nn.init.normal_(self.embedding.weight, mean=0, std=emb_dim ** -0.5)
        #     nn.init.constant_(self.embedding.weight[padding_idx], 0)

        self.LayerNorm = nn.LayerNorm(emb_dim, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(input_size=emb_dim, hidden_size=hidden_size, bidirectional=bidirectional,
                          batch_first=True)

    def forward(self, input):
        emb = self.embedding(input)
        emb = self.LayerNorm(emb)
        emb = self.dropout(emb)

        output, hidden = self.rnn(emb)
        # if self.bidirectional:
        #     output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]

        return output, hidden


if __name__ == '__main__':
    enc = RNNEncoder(12)
    print(enc.embedding.weight.requires_grad)
