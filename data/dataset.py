"""
# dataset.py created by bisheng at 2020/5/31 22:43.
"""

import torch
import torch.utils.data
import numpy as np


def paired_collate_fn(insts):
    graph_spe, adj, diff, node_diff, question_spe, question_full, answer, graph_node_mask, graph_full, question2full_idx2idx = list(
        zip(*insts))
    return (graph_spe, adj, diff, node_diff, question_spe, question_full, answer, graph_node_mask, graph_full,
            question2full_idx2idx)


def collate_fn(insts):
    ''' Pad the instance to the max seq length in batch '''

    max_len = max(len(inst) for inst in insts) + 2

    batch_seq = np.array([
        [2] + inst + [3] + [0] * (max_len - len(inst))
        for inst in insts])

    batch_seq = torch.LongTensor(batch_seq)

    return batch_seq


class QG_and_QA_Dataset(torch.utils.data.Dataset):
    def __init__(self, graph_word2idx, question_word2idx, rel_word2idx, full_word2idx, graph_spe, graph_full,
                 graph_spe_seq, graph_full_seq, adj,
                 question_spe, question_full, question_spe_e, question_full_e, answer, answer_seq, graph_node_mask_seq,
                 relation, graph_node_mask):
        self._graph_word2idx = graph_word2idx
        self._graph_idx2word = {idx: word for word, idx in graph_word2idx.items()}

        self._question_word2idx = question_word2idx
        self._question_idx2word = {idx: word for word, idx in question_word2idx.items()}

        self._full_word2idx = full_word2idx
        self._full_idx2word = {idx: word for word, idx in full_word2idx.items()}

        self._rel_word2idx = rel_word2idx
        self._rel_idx2word = {idx: word for word, idx in rel_word2idx.items()}

        self._graph_spe = graph_spe
        self._graph_full = graph_full
        self._graph_spe_seq = graph_spe_seq
        self._graph_full_seq = graph_full_seq
        self._adj = adj

        self._question_spe = question_spe
        self._question_full = question_full

        self._question_spe_e = question_spe_e
        self._question_full_e = question_full_e

        self._answer = answer
        self._answer_seq = answer_seq
        self._graph_node_mask = graph_node_mask
        self._graph_node_mask_seq = graph_node_mask_seq

        self._relation = relation

        self._full2question_idx2idx = torch.LongTensor([0] * len(self._full_idx2word))
        for key in self._full_idx2word.keys():
            self._full2question_idx2idx[key] = self._question_word2idx.get(self._full_idx2word[key],
                                                                           self._question_word2idx['<unk>'])
        self._question2full_idx2idx = torch.LongTensor([0] * len(self._question_idx2word))
        for key in self._question_idx2word.keys():
            self._question2full_idx2idx[key] = self._full_word2idx[self._question_idx2word[key]]

    @property
    def n_insts(self):
        ''' Property for dataset size '''
        return len(self._graph_spe)

    @property
    def src_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._graph_word2idx)

    @property
    def tgt_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._question_word2idx)

    @property
    def full_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._full_word2idx)

    @property
    def rel_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._rel_word2idx)

    @property
    def src_word2idx(self):
        ''' Property for word dictionary '''
        return self._graph_word2idx

    @property
    def tgt_word2idx(self):
        ''' Property for word dictionary '''
        return self._question_word2idx

    @property
    def full_word2idx(self):
        ''' Property for word dictionary '''
        return self._full_word2idx

    @property
    def rel_word2idx(self):
        ''' Property for word dictionary '''
        return self._rel_word2idx

    @property
    def full_idx2word(self):
        ''' Property for index dictionary '''
        return self._full_idx2word

    @property
    def src_idx2word(self):
        ''' Property for index dictionary '''
        return self._graph_idx2word

    @property
    def tgt_idx2word(self):
        ''' Property for index dictionary '''
        return self._question_idx2word

    @property
    def rel_idx2word(self):
        ''' Property for index dictionary '''
        return self._rel_idx2word

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        if self._question_spe is not None:
            return self._graph_spe[idx], self._adj[idx], self._question_spe[idx], self._question_full[idx], \
                   self._answer[idx], self._graph_node_mask[idx], self._graph_full[idx], self._graph_spe_seq[idx], \
                   self._question_spe_e[idx], self._question_full_e[idx], self._answer_seq[idx], \
                   self._graph_node_mask_seq[idx], self._graph_full_seq[idx], self._relation[
                       idx], self._question2full_idx2idx
        return self._graph_spe[idx], self._adj[idx], self._question_full[idx], self._answer[idx], self._graph_node_mask[
            idx], self._graph_full[idx], self._graph_spe_seq[idx], self._question_spe_e[idx], self._question_full_e[
                   idx], self._answer_seq[idx], self._graph_node_mask_seq[idx], self._graph_full_seq[idx], \
               self._relation[idx], self._question2full_idx2idx


class Question_Generation_Dataset(torch.utils.data.Dataset):
    def __init__(self, graph_word2idx, question_word2idx, full_word2idx, graph_spe, graph_full, adj, diff, node_diff,
                 question_spe, question_full, answer, graph_node_mask):
        self._graph_word2idx = graph_word2idx
        self._graph_idx2word = {idx: word for word, idx in graph_word2idx.items()}

        self._question_word2idx = question_word2idx
        self._question_idx2word = {idx: word for word, idx in question_word2idx.items()}

        self._full_word2idx = full_word2idx
        self._full_idx2word = {idx: word for word, idx in full_word2idx.items()}

        self._graph_spe = graph_spe
        self._graph_full = graph_full
        self._adj = adj
        self._diff = diff
        self._node_diff = node_diff
        self._question_spe = question_spe
        self._question_full = question_full
        self._answer = answer
        self._graph_node_mask = graph_node_mask

        self._question2full_idx2idx = torch.LongTensor([0] * len(self._question_idx2word))
        for key in self._question_idx2word.keys():
            self._question2full_idx2idx[key] = self._full_word2idx[self._question_idx2word[key]]

    @property
    def n_insts(self):
        ''' Property for dataset size '''
        return len(self._graph_spe)

    @property
    def src_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._graph_word2idx)

    @property
    def tgt_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._question_word2idx)

    @property
    def full_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._full_word2idx)

    @property
    def src_word2idx(self):
        ''' Property for word dictionary '''
        return self._graph_word2idx

    @property
    def tgt_word2idx(self):
        ''' Property for word dictionary '''
        return self._question_word2idx

    @property
    def full_word2idx(self):
        ''' Property for word dictionary '''
        return self._full_word2idx

    @property
    def full_idx2word(self):
        ''' Property for index dictionary '''
        return self._full_idx2word

    @property
    def src_idx2word(self):
        ''' Property for index dictionary '''
        return self._graph_idx2word

    @property
    def tgt_idx2word(self):
        ''' Property for index dictionary '''
        return self._question_idx2word

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        if self._question_spe is not None:
            return self._graph_spe[idx], self._adj[idx], self._diff[idx], self._node_diff[idx], self._question_spe[idx], \
                   self._question_full[idx], self._answer[idx], self._graph_node_mask[idx], self._graph_full[
                       idx], self._question2full_idx2idx
        return self._graph_spe[idx], self._adj[idx], self._diff[idx], self._node_diff[idx], self._answer[idx], \
               self._graph_node_mask[idx], self._graph_full[idx], self._question2full_idx2idx
