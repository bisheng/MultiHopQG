# -*- coding: utf-8 -*-

""" 
Created at 2019-06-12 21:17:47
==============================
gru_self 
"""

# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
from qg_qa.modules.attention import Attention
import numpy as np


class GRU_SELF(nn.Module):

    def __init__(self, emb_dim, hidden_size, output_size):
        super(GRU_SELF, self).__init__()
        self.attention = Attention(hidden_size)
        self.Wr = nn.Parameter(torch.FloatTensor(emb_dim, hidden_size), requires_grad=True)
        nn.init.xavier_normal_(self.Wr)
        self.Wu = nn.Parameter(torch.FloatTensor(emb_dim, hidden_size), requires_grad=True)
        nn.init.xavier_normal_(self.Wu)

        self.Ur = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size), requires_grad=True)
        nn.init.xavier_normal_(self.Ur)
        self.Uu = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size), requires_grad=True)
        nn.init.xavier_normal_(self.Uu)

        self.Cr = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size), requires_grad=True)
        nn.init.xavier_normal_(self.Cr)
        self.Cu = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size), requires_grad=True)
        nn.init.xavier_normal_(self.Cu)

        self.W = nn.Parameter(torch.FloatTensor(emb_dim, hidden_size), requires_grad=True)
        nn.init.xavier_normal_(self.W)
        self.C = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size), requires_grad=True)
        nn.init.xavier_normal_(self.C)
        self.U = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size), requires_grad=True)
        nn.init.xavier_normal_(self.U)

        self.p_gen = nn.Linear(2 * hidden_size + emb_dim, 1)
        self.temperature = np.power(hidden_size, 0.5)
        self.fc_out = nn.Linear(2 * hidden_size + emb_dim, output_size)

    def forward(self, emb, hidden, encoder_outputs, mask=None):
        '''

        :param emb: b*l*emb_dim
        :param hidden: b*1*h
        :param encoder_outputs: b*N*h
        :param c_i: b*1*h
        :param diff: b*1*h
        :param mask: b*1*N
        :return:
        '''
        outputs = None
        atts = None
        p_gen = None
        for i in range(emb.size(1)):
            c_j, att = self.attention(hidden, encoder_outputs, mask=mask)  # , gen
            w = emb[:, i:i + 1, :]  # b*1*emb_dim
            r_j = torch.sigmoid(
                torch.matmul(w, self.Wr) + torch.matmul(hidden, self.Ur) + torch.matmul(c_j, self.Cr))
            u_j = torch.sigmoid(
                torch.matmul(w, self.Wu) + torch.matmul(hidden, self.Uu) + torch.matmul(c_j, self.Cu))
            s_j_t = torch.tanh(
                torch.matmul(w, self.W) + torch.matmul(hidden * r_j, self.U) + torch.matmul(c_j, self.C))
            hidden = (1 - u_j) * hidden + u_j * s_j_t
            gen_prediction = self.fc_out(torch.cat((hidden, c_j, w), dim=-1))
            if outputs is None:
                outputs = gen_prediction
            else:
                outputs = torch.cat((outputs, gen_prediction), 1)

            # att = torch.bmm(hidden + c_j, encoder_outputs.transpose(1, 2))
            # att = att / self.temperature
            # if mask is not None:
            #     att.data.masked_fill_(mask, -10000.0)  # -float('inf')

            if atts is None:
                atts = att
            else:
                atts = torch.cat((atts, att), 1)

            gen = self.p_gen(torch.cat((hidden, c_j, w), dim=-1)).squeeze(2)
            gen=F.sigmoid(gen)
            if p_gen is None:
                p_gen = gen
            else:
                p_gen = torch.cat((p_gen, gen), 1)

        return outputs, hidden, atts, p_gen
