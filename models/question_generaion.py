import torch
import torch.nn as nn
from models.encoder import GAT_Transformer_Encoder
from models.decoder import Decoder_Copy

class QG(nn.Module):
    def __init__(self, node_vocab_size, q_vocab_size, full_vocab_size, max_len, e_emb_dim=256, q_emb_dim=512, hidden_size=512, sos_id=0, nheads=3, alpha=0.2, dropout=0.1, use_copy=True,
                 encoder_type='GNN'):
        super(QG, self).__init__()
        self.encoder_type = encoder_type

        self.node_embedding = nn.Embedding(node_vocab_size, e_emb_dim)
        self.ans_embedding = nn.Embedding(2, e_emb_dim)

        if self.encoder_type == 'GNN':
            self.encoder = GAT_Transformer_Encoder(nfeat=e_emb_dim, nhid=hidden_size, dropout=dropout, alpha=alpha,
                                                   nheads=nheads)
        else:
            pass
        self.decoder = Decoder_Copy(q_vocab_size=q_vocab_size, full_vocab_size=full_vocab_size,
                                    max_len=max_len, sos_id=sos_id,
                                    hidden_size=hidden_size,
                                    q_emb_dim=q_emb_dim, dropout=dropout, use_copy=use_copy)


    def forward(self, nodes, ans, mask, qidx2full, nidx2full, adj=None, question=None, teacher_forcing_ratio=0):
        r'''

        Args:
            nodes: (b,N)
            ans: (b,N)
            n_diff: (b,N)
            q_diff: (b,1)
            adj: (b,N,N)
            mask: (b,1,N)
        Examples::
            >>>
            >>>
        '''
        nodes_emb = self.node_embedding(nodes)  # (b,N,emb_dim)
        ans_emb = self.ans_embedding(ans)  # (b,N,emb_dim)

        features = nodes_emb + ans_emb
        if self.encoder_type == 'GNN':
            assert adj is not None
            encoder_output = self.encoder(features, adj)  # (b,N,hidden)
            hidden=encoder_output[:, 0, :].unsqueeze(1)
        elif self.encoder_type == 'RNN':
            encoder_output = None
            hidden = encoder_output[:, -1, :].unsqueeze(1)
        elif self.encoder_type == 'Transformer':
            encoder_output = None
            hidden = encoder_output.sum(1).unsqueeze(1)
        else:
            encoder_output = None
            hidden = None
            print('Error Type of Encoder')


        mask = mask.eq(1).unsqueeze(1)
        # print(encoder_output.size(),hidden.size(),mask.size(),qidx2full.size(),nidx2full.size(),question.size())
        decoder_output, decoder_hidden = self.decoder(encoder_outputs=encoder_output, hidden=hidden,
                                                      attention_mask=mask, qidx2full=qidx2full, nidx2full=nidx2full,
                                                      inputs=question, teacher_forcing_ratio=teacher_forcing_ratio)
        return decoder_output, decoder_hidden
