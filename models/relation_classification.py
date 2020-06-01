import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, source_vocab_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.input_size = source_vocab_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.dropout = 0.1

        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.num_layers, dropout=self.dropout, bidirectional=True,
                          batch_first=True)

    def forward(self, input_seqs, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        # input: S*B
        embedded = self.embedding(input_seqs)  # S*B*D
        # packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # print(embedded.size())
        outputs, hidden = self.gru(embedded, hidden)
        # outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # outputs: S*B*2D
        # hidden: 2*B*D
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        hidden = hidden[:1, :, :] + hidden[-1:, :, :]
        # outputs: S*B*D
        # hidden: 1*B*D
        return outputs, hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, target_vocab_size):
        super(DecoderRNN, self).__init__()
        # Define parameters
        self.hidden_size = hidden_size
        self.output_size = target_vocab_size
        self.num_layers = 1
        self.dropout_p = 0.1
        # Define layers
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.num_layers, dropout=self.dropout_p, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, word_input, prev_hidden):
        # Get the embedding of the current input word (last output word)
        # word input: B
        # prev_hidden: 1*B*D
        embedded = self.embedding(word_input)  # B*D
        embedded = self.dropout(embedded)

        rnn_output, hidden = self.gru(embedded, prev_hidden)
        # rnn_output : 1*B*D
        # hidden : 1*B*D
        rnn_output = rnn_output.squeeze(0)  # B*D
        output = self.out(rnn_output)  # B*target_vocab_size
        return output, hidden


class Relation_classification(nn.Module):
    def __init__(self, source_vocab_size, hidden_size, target_vocab_size):
        super(Relation_classification, self).__init__()
        self.encoder = EncoderRNN(source_vocab_size, hidden_size)
        self.decoder = DecoderRNN(hidden_size, target_vocab_size)

    def forward(self, input_seqs, word_input):
        _, hidden = self.encoder(input_seqs)
        output, hidden = self.decoder(word_input, hidden)
        output = F.softmax(output, dim=-1)
        return output, hidden
