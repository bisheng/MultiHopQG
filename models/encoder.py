import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import GraphAttentionLayer,PositionwiseFeedForward

class GAT_Transformer_Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        super(GAT_Transformer_Encoder, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.pos_ffn = PositionwiseFeedForward(nhid * nheads, nhid, dropout=dropout)
        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False)
        self.pos_ffn2 = PositionwiseFeedForward(nhid, nhid, dropout=dropout)

    def forward(self, x, adj):
        r'''

        Args:
            x: (b,N,emb_dim)
            adj: (b,N,N)
        '''
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.pos_ffn(x)
        x = self.out_att(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.pos_ffn2(x)
        return x

class RNN_Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout, alpha, nheads):
        super(GAT_Transformer_Encoder, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.pos_ffn = PositionwiseFeedForward(nhid * nheads, nhid, dropout=dropout)
        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False)
        self.pos_ffn2 = PositionwiseFeedForward(nhid, nhid, dropout=dropout)

    def forward(self, x, adj):
        r'''

        Args:
            x: (b,N,emb_dim)
            adj: (b,N,N)
        '''
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.pos_ffn(x)
        x = self.out_att(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.pos_ffn2(x)
        return x