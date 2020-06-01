import torch
import torch.nn as nn
import torch.nn.functional as F
from .gru_self import GRU_SELF
import random


class Decoder_Copy(nn.Module):
    def __init__(self, q_vocab_size, full_vocab_size, max_len, sos_id, q_emb_dim=512,
                 hidden_size=512, dropout=0.1, use_copy=True):
        super(Decoder_Copy, self).__init__()

        self.embedding = nn.Embedding(q_vocab_size, q_emb_dim)
        self.sos_id = sos_id
        self.max_length = max_len

        self.rnn = GRU_SELF(q_emb_dim, hidden_size,q_vocab_size)

        self.output_size = full_vocab_size
        self.use_copy = use_copy

        self.hidden_size = hidden_size
        self.init_input = None
        self.input_dropout = nn.Dropout(p=dropout)

        # self.out = nn.Linear(hidden_size, q_vocab_size)
        self.out = nn.LogSoftmax(dim=-1)

    def forward_step(self, input_var, hidden, encoder_outputs, mask=None):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)

        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        output, hidden, atts, p_gen = self.rnn(embedded, hidden, encoder_outputs, mask)

        predicted_softmax = self.out(output)
        return predicted_softmax, hidden, atts, p_gen

    def forward(self, encoder_outputs, qidx2full=None, nidx2full=None, inputs=None, hidden=None,teacher_forcing_ratio=0, attention_mask=None):
        r'''

        args:
            inputs: (b,l)
            encoder_outputs: (b,N,hidden)
            hidden: (b,1,hidden)
            qidx2full: (b,V_q)
            nidx2full: (b,N)
            attention_mask: (b,1,N)
        '''

        batch_size = encoder_outputs.size(0)
        if inputs is None:
            if teacher_forcing_ratio > 0:
                teacher_forcing_ratio = 0
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1).to(encoder_outputs.device)
            max_length = self.max_length
        else:
            max_length = inputs.size(1)

        if hidden is None:
            hidden = encoder_outputs.sum(1).unsqueeze(1)
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = None

        def decode(step_output, step_attn, decoder_outputs, qidx2full, nidx2full, use_copy, output_size, gen):
            if use_copy:
                vocab_dist = torch.zeros((step_output.size(0), output_size)).to(step_output.device)
                vocab_dist = vocab_dist.scatter_add_(1, qidx2full, step_output * gen)
                vocab_dist = vocab_dist.scatter_add_(1, nidx2full, step_attn * (1 - gen))
            else:
                vocab_dist = step_output
            if decoder_outputs is None:
                decoder_outputs = vocab_dist.unsqueeze(1)
            else:
                decoder_outputs = torch.cat((decoder_outputs, vocab_dist.unsqueeze(1)), 1)
            return decoder_outputs

        if use_teacher_forcing:
            decoder_input = inputs
            decoder_output, hidden, attn, p_gen = self.forward_step(decoder_input, hidden, encoder_outputs,mask=attention_mask)
            for di in range(decoder_output.size(1)):
                step_output = decoder_output[:, di, :]
                if attn is not None:
                    step_attn = attn[:, di, :]
                else:
                    step_attn = None
                gen = p_gen[:, di:di + 1]
                decoder_outputs = decode(step_output, step_attn, decoder_outputs, qidx2full, nidx2full, self.use_copy,
                                         self.output_size, gen)
        else:
            decoder_input = inputs[:, 0].unsqueeze(1)
            for di in range(max_length):
                decoder_output, hidden, step_attn, gen = self.forward_step(decoder_input, hidden, encoder_outputs, mask=attention_mask)
                step_output = decoder_output.squeeze(1)
                step_attn = step_attn.squeeze(1)
                decoder_input = step_output.topk(1)[1]
                decoder_outputs = decode(step_output, step_attn, decoder_outputs, qidx2full, nidx2full, self.use_copy,
                                         self.output_size, gen)

        return decoder_outputs, hidden
