# -*- coding: utf-8 -*-
"""
Created by fx at 2019-10-03 10:37:34
==============================
graph_bulding
"""

import tagme
import re

import nltk
from nltk.tokenize import word_tokenize

import numpy as np
import scipy.sparse as sp

tagme.GCUBE_TOKEN = "37a8295d-8ffc-4c7d-bfe9-4f9bb474cef9-843339462"

import networkx as nx

from torchtext import data
import torch
import json

from wikidata.client import Client

import wikipediaapi

import random
import eventlet
import time

wiki = wikipediaapi.Wikipedia('en')

client = Client()
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


def final_preprocess_data(question, question_e, relations, answers, triples, nodes_len=None, question_len=None,
                          relation_len=None, sequence_triple_len=None):
    graph_triples = []
    graph_nodes = []
    entity_set = []
    sequence_triple = []
    for triple in triples:
        subject = re.sub('\(.*?\)', '', triple[0]).strip().lower()
        rel = triple[1]
        object = re.sub('\(.*?\)', '', triple[2]).strip().lower()
        rel_inv = triple[1] + '_inv'
        sequence_triple.extend([subject, rel, object])
        graph_nodes.extend([subject, rel, object, rel_inv])
        graph_triples.extend([[subject, rel, object], [object, rel_inv, subject]])
        entity_set.extend([subject, object])

    entity_set = list(set(entity_set))
    graph_nodes = ['global vertex'] + list(set(graph_nodes))
    g_len = len(graph_nodes)
    if nodes_len is not None:
        if len(graph_nodes) >= nodes_len:
            graph_nodes = graph_nodes[:nodes_len]
        else:
            graph_nodes = graph_nodes + ['<pad>'] * (nodes_len - len(graph_nodes))
    else:
        print(len(graph_nodes))

    if sequence_triple_len is not None:
        if len(sequence_triple) >= sequence_triple_len:
            sequence_triple = sequence_triple[:sequence_triple_len]
        else:
            sequence_triple = sequence_triple + ['<pad>'] * (sequence_triple_len - len(sequence_triple))

    mask = []
    for gn in graph_nodes:
        if gn in entity_set:
            mask.append(0)
        else:
            mask.append(1)

    sequence_mask = []
    for gn in sequence_triple:
        if gn in entity_set:
            sequence_mask.append(0)
        else:
            sequence_mask.append(1)

    answer = [re.sub('\(.*?\)', '', a).strip().lower() for a in answers]
    answer_encode = [1 if e in answer else 0 for e in graph_nodes]
    sequence_answer = [1 if e in answer else 0 for e in sequence_triple]

    adj = np.zeros((len(graph_nodes), len(graph_nodes)))

    for entity in entity_set:
        try:
            adj[graph_nodes.index('global vertex'), graph_nodes.index(entity)] = 1
            # adj[graph_nodes.index(entity), graph_nodes.index('global vertex')] = 1
        except:
            pass
    for triple in graph_triples:
        try:
            adj[graph_nodes.index(triple[0]), graph_nodes.index(triple[1])] = 1
            adj[graph_nodes.index(triple[1]), graph_nodes.index(triple[2])] = 1
        except:
            pass
    adj = sp.coo_matrix(adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    question_words = ['<sos>'] + question + ['<eos>']
    question_words_e = ['<sos>'] + question_e + ['<eos>']

    q_len = len(question_words)
    if question_len is not None:
        if len(question_words) >= question_len:
            # mask = [0] * question_len
            question_words = question_words[:question_len]
            question_words_e = question_words_e[:question_len]
        else:
            # mask = [0] * len(graph_nodes) + [1] * (nodes_len - len(graph_nodes))
            question_words = question_words + ['<pad>'] * (question_len - len(question_words))
            question_words_e = question_words_e + ['<pad>'] * (question_len - len(question_words_e))
    else:
        print(len(question_words))

    relations = ['<sos>'] + relations + ['<eos>']
    r_len=len(relations)
    if relation_len is not None:
        if len(relations) >= relation_len:
            relations = relations[:relation_len]
        else:
            relations = relations + ['<pad>'] * (relation_len - len(relations))

    return graph_nodes, answer_encode, adj, question_words, question_words_e, relations, mask, g_len, q_len, sequence_triple, sequence_answer, sequence_mask,r_len


def get_mask(triples, graph_nodes):
    entity_set = []
    for triple in triples:
        subject = re.sub('\(.*?\)', '', triple[0]).strip().lower()  # triple[0]
        object = re.sub('\(.*?\)', '', triple[2]).strip().lower()  # triple[2]
        entity_set.extend([subject, object])

    entity_set = list(set(entity_set))

    mask = []
    for gn in graph_nodes:
        if gn in entity_set:
            mask.append(0)
        else:
            mask.append(1)

    return mask


def get_final_preprocess_data(complex_data_path=None, simple_data_path=None):
    all_data = []

    # fb2str = get_fb2str_dict()
    # with open('../data/subgraph.txt', 'r', encoding='utf-8') as f:
    #     triples = ['<t>'.join(line.strip().split('<t>')[:-1]) for line in f.readlines()]
    #
    # temp_help_list=[]
    # for triple in triples:
    #     t_list = triple.split('<t>')
    #     triple = []
    #     for tt in t_list:
    #         es = tt.split(' ')
    #         triple.append([re.sub('\(.*?\)', '', fb2str.get(e, e)).strip().lower() for e in es])
    #     temp_help_list.append('<t>'.join(['[t]'.join(t) for t in triple]))


    if complex_data_path:
        with open(complex_data_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                info = line.strip().split('\t')
                question = info[1].split('<t>')
                question_e = info[2].split('<t>')
                relations = info[3].split('<t>')
                answers = info[4].split('<t>')

                # triples = [t.split('[t]') for t in temp_help_list[int(info[0])].split('<t>')]
                triples = [t.split('[t]') for t in info[5].split('<t>')]
                all_data.append([question, question_e, relations, answers, triples])

    if simple_data_path:
        with open(simple_data_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                info = line.strip().split('\t')
                question = info[1].split('<t>')
                question_e = info[2].split('<t>')
                relations = info[3].split('<t>')
                answers = info[4].split('<t>')
                triples = [t.split('[t]') for t in info[5].split('<t>')]
                all_data.append([question, question_e, relations, answers, triples])

    print(len(all_data))
    train_data, test_data = all_data[:int(len(all_data) * 0.9)], all_data[int(len(all_data) * 0.9):]
    print(len(train_data),len(test_data))
    print(train_data[0])

    train_graphs = []
    train_answers = []
    train_adjs = []
    train_questions = []
    train_questions_e = []
    train_masks = []
    train_relations = []

    train_seq_triple = []
    train_seq_ans = []
    train_seq_mask = []

    q_lens = 0
    g_lens = 0
    r_lens=0
    cc = 0
    n_handle = []

    for data in train_data:
        graph_nodes, answer_encode, adj, question_words, question_words_e, relation, mask, g_len, q_len, sequence_triple, sequence_answer, sequence_mask,r_len = final_preprocess_data(
            data[0],
            data[1],
            data[2],
            data[3],
            data[4],
            nodes_len=50,
            question_len=25,
            relation_len=5,
            sequence_triple_len=50)
        train_graphs.append(graph_nodes)
        train_answers.append(answer_encode)
        train_adjs.append(adj.todense())
        train_questions_e.append(question_words_e)
        train_questions.append(question_words)
        train_masks.append(mask)
        train_relations.append(relation)

        train_seq_triple.append(sequence_triple)
        train_seq_ans.append(sequence_answer)
        train_seq_mask.append(sequence_mask)

        q_lens += q_len
        g_lens += g_len
        r_lens+=r_len
        cc += 1
        if cc % 100 == 1:
            print(cc)

    print(q_lens / len(train_graphs), g_lens / len(train_graphs),r_lens/len(train_graphs))

    test_graphs = []
    test_answers = []
    test_adjs = []
    test_questions_e = []
    test_questions = []
    test_relations = []
    test_masks = []

    test_seq_triple = []
    test_seq_ans = []
    test_seq_mask = []

    for data in test_data:
        graph_nodes, answer_encode, adj, question_words, question_words_e, relation, mask, g_len, q_len,sequence_triple, sequence_answer, sequence_mask,r_len = final_preprocess_data(
            data[0],
            data[1],
            data[2],
            data[3],
            data[4],
            nodes_len=50,
            question_len=25,
            relation_len=5,
            sequence_triple_len=50)

        test_graphs.append(graph_nodes)
        test_answers.append(answer_encode)
        test_adjs.append(adj.todense())
        test_questions_e.append(question_words_e)
        test_questions.append(question_words)
        test_relations.append(relation)
        test_masks.append(mask)

        test_seq_triple.append(sequence_triple)
        test_seq_ans.append(sequence_answer)
        test_seq_mask.append(sequence_mask)

    graph_word2idx, question_word2idx, full_word2idx = bulid_vocab_graph_and_question(train_graphs + test_graphs,
                                                                                      train_questions_e + test_questions_e)
    rel2idx=bulid_relation_vocab(train_relations+test_relations)

    print(len(question_word2idx))
    print(len(graph_word2idx), len(question_word2idx), len(full_word2idx))

    train_graphs_spe = convert_instance_to_idx_seq(train_graphs, graph_word2idx)
    train_graphs_full = convert_instance_to_idx_seq(train_graphs, full_word2idx)

    train_graphs_spe_seq = convert_instance_to_idx_seq(train_seq_triple, graph_word2idx)
    train_graphs_full_seq = convert_instance_to_idx_seq(train_seq_triple, full_word2idx)

    train_questions_spe = convert_instance_to_idx_seq(train_questions, question_word2idx)
    train_questions_full = convert_instance_to_idx_seq(train_questions, full_word2idx)

    train_questions_spe_e = convert_instance_to_idx_seq(train_questions_e, question_word2idx)
    train_questions_full_e = convert_instance_to_idx_seq(train_questions_e, full_word2idx)

    train_relations=convert_instance_to_idx_seq(train_relations,rel2idx)


    test_graphs_spe = convert_instance_to_idx_seq(test_graphs, graph_word2idx)
    test_graphs_full = convert_instance_to_idx_seq(test_graphs, full_word2idx)

    test_graphs_spe_seq = convert_instance_to_idx_seq(test_seq_triple, graph_word2idx)
    test_graphs_full_seq = convert_instance_to_idx_seq(test_seq_triple, full_word2idx)

    test_questions_spe = convert_instance_to_idx_seq(test_questions, question_word2idx)
    test_questions_full = convert_instance_to_idx_seq(test_questions, full_word2idx)

    test_questions_spe_e = convert_instance_to_idx_seq(test_questions_e, question_word2idx)
    test_questions_full_e = convert_instance_to_idx_seq(test_questions_e, full_word2idx)

    test_relations = convert_instance_to_idx_seq(test_relations, rel2idx)

    data = {
        'dict': {
            'graph_node': graph_word2idx,
            'question_words': question_word2idx,
            'full': full_word2idx,
            'relation':rel2idx},
        'train': {
            'graph_spe': torch.LongTensor(train_graphs_spe),
            'graph_full': torch.LongTensor(train_graphs_full),
            'graph_spe_seq': torch.LongTensor(train_graphs_spe_seq),
            'graph_full_seq': torch.LongTensor(train_graphs_full_seq),
            'adj': torch.FloatTensor(np.array(train_adjs)),
            'question_spe': torch.LongTensor(train_questions_spe),
            'question_full': torch.LongTensor(train_questions_full),
            'question_spe_e': torch.LongTensor(train_questions_spe_e),
            'question_full_e': torch.LongTensor(train_questions_full_e),
            'answer': torch.LongTensor(train_answers),
            'graph_node_mask': torch.LongTensor(train_masks),
            'answer_seq': torch.LongTensor(train_seq_ans),
            'relation':torch.LongTensor(train_relations),
            'graph_node_mask_seq': torch.LongTensor(train_seq_mask)},
        'test': {
            'graph_spe': torch.LongTensor(test_graphs_spe),
            'graph_full': torch.LongTensor(test_graphs_full),
            'graph_spe_seq': torch.LongTensor(test_graphs_spe_seq),
            'graph_full_seq': torch.LongTensor(test_graphs_full_seq),
            'adj': torch.FloatTensor(np.array(test_adjs)),
            'question_spe': torch.LongTensor(test_questions_spe),
            'question_full': torch.LongTensor(test_questions_full),
            'question_spe_e': torch.LongTensor(test_questions_spe_e),
            'question_full_e': torch.LongTensor(test_questions_full_e),
            'answer': torch.LongTensor(test_answers),
            'graph_node_mask': torch.LongTensor(test_masks),
            'answer_seq': torch.LongTensor(test_seq_ans),
            'relation': torch.LongTensor(test_relations),
            'graph_node_mask_seq': torch.LongTensor(test_seq_mask)}}
    torch.save(data, '../data/complex_data.pth')

def bulid_relation_vocab(relations):
    rel2idx={'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
    all_rel=set(r for relation in relations for r in relation)
    for r in all_rel:
        if r not in rel2idx.keys():
            rel2idx[r]=len(rel2idx)
    # rel2idx['<unk>']=len(rel2idx)
    return rel2idx

def bulid_vocab_graph_and_question(graph_nodes, question_words):
    graph_word2idx = {'<pad>': 0, '<unk>': 1}

    question_word2idx = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}

    full_word2idx = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}

    src_set = set(node for graph in graph_nodes for node in graph)
    tgt_set = set(w for question in question_words for w in question)
    full_set = src_set | tgt_set

    for word in src_set:
        if word not in graph_word2idx.keys():
            graph_word2idx[word] = len(graph_word2idx)

    for word in tgt_set:
        if word not in question_word2idx.keys():
            question_word2idx[word] = len(question_word2idx)

    for word in full_set:
        if word not in full_word2idx.keys():
            full_word2idx[word] = len(full_word2idx)

    word_count = {}
    for graph in graph_nodes:
        for node in graph:
            if node in word_count.keys():
                word_count[node] += 1
            else:
                word_count[node] = 1
    for question in question_words:
        for w in question:
            if w in word_count.keys():
                word_count[w] += 1
            else:
                word_count[w] = 1
    cc = 0
    for w in word_count.keys():
        if word_count[w] < 2:
            cc += 1
    print(cc)

    return graph_word2idx, question_word2idx, full_word2idx


def convert_instance_to_idx_seq(word_insts, word2idx):
    ''' Mapping words to idx sequence. '''

    return [[word2idx.get(w, word2idx['<unk>']) for w in s] for s in word_insts]



def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))

    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def build_vocab(src_word_insts, tgt_word_insts):
    src_word2idx = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
    tgt_word2idx = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
    full_word2idx = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}

    src_set = set(w for sent in src_word_insts for w in sent)
    tgt_set = set(w for sent in tgt_word_insts for w in sent)
    full_set = src_set | tgt_set

    for word in src_set:
        src_word2idx[word] = len(src_word2idx)
    for word in tgt_set:
        tgt_word2idx[word] = len(tgt_word2idx)
    for word in full_set:
        full_word2idx[word] = len(full_word2idx)

    return src_word2idx, tgt_word2idx, full_word2idx


def preprocess_dict(fb2w_path):
    '''
    实现从freebase ID到string的映射，暂时使用已处理好的fb2str.txt。
    :param fb2w_path:
    :return:
    '''
    fb2w = get_intermediate_fb2w_dict(fb2w_path)
    fb2str = fb2w
    count = 0
    for key in fb2w.keys():
        try:
            fb2str[key] = str(client.get(fb2w[key], load=True).label).lower()
            count += 1
            if count % 100 == 1:
                print(count)
        except:
            del fb2str[key]
    with open('fb2str_dict_temp.txt', 'w', encoding='utf-8') as f:
        for key in fb2str.keys():
            f.write(key + '\t' + fb2str[key] + '\n')
    torch.save({'fb2str': fb2str}, 'fb2str_all.txt')


def get_fb2str_dict(fb2str_path='../data/fb2str_final.txt'):
    with open(fb2str_path, 'r', encoding='utf-8') as f:
        fb2str = {line.strip().split('\t')[0]: line.strip().split('\t')[3] for line in f.readlines() if
                  line.strip() != ''}
    return fb2str


def get_entity_and_relation(question, triples, answer):
    entity_set = []
    for triple in triples:
        entity_set.extend([triple[0], triple[2]])
    entity_set = list(set(entity_set))

    mentions = {}
    mention_entity = []
    for i, ent in enumerate(tagme.annotate(question, lang='en').get_annotations(0.1)):
        for e in entity_set:
            if e not in answer and ent.entity_title in e or ent.mention in e.lower():
                mention_entity.append(e)
                mentions['e_%s' % (i)] = re.sub('\(.*?\)', '', e).strip().lower()
                question = question.replace(ent.mention, 'e_%s' % (i))

    question_e = word_tokenize(question)
    question_e = '<t>'.join(question_e)
    question = question_e
    for key in mentions.keys():
        '''
        这里需要继续优化，主要为了解决copy，问题中mention和triples中实体表示应该对应
        '''
        question = question.replace(key, mentions[key])

    print(mention_entity)
    if len(mention_entity) > 0:
        relations = []

        def help(mention, res,temps):
            ts=temps[:]
            for t in ts:
                if t[0] == mention:
                    ts.remove(t)
                    res = res + [t[1]] if t[2] in answer else help(t[2], res + [t[1]],ts)
                    if res:
                        return res
            ts = temps[:]
            for t in ts:
                if t[2]==mention:
                    ts.remove(t)
                    res=res+[t[1]] if t[0] in answer else help(t[0],res+[t[1]],ts)
                    if res:
                        return res
            return []

        for men in mention_entity:
            temp=triples[:]
            relations = help(men, [],temp)
            if relations:
                break
        if len(relations) > 0:
            return question, question_e, '<t>'.join(answer), '<t>'.join(relations), '<t>'.join(
                ['[t]'.join([re.sub('\(.*?\)', '', t).strip().lower() for t in triple]) for triple in triples])
    return None, None, None, None, None


def prepare_complex_question(question_path='../data/full.tgt', subgraph_path='../data/subgraph.txt',
                             ans_path='../data/train.src'):
    with open(question_path, 'r', encoding='utf-8') as f:
        questions = [line.strip().lower() for line in f.readlines()]
    with open(subgraph_path, 'r', encoding='utf-8') as f:
        triples = ['<t>'.join(line.strip().split('<t>')[:-1]) for line in f.readlines()]
    with open(ans_path, 'r', encoding='utf-8') as f:
        ans = [line.strip().replace('￨H', '').replace('￨E', '') for line in f.readlines()]

    assert len(questions) == len(triples)
    assert len(questions) == len(ans)

    handel_set = []
    try:
        with open('../data/complex_data_webquestions.txt', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if line.strip():
                    handel_set.append(int(line.strip().split('\t')[0]))
    except:
        handel_set = []
    print(handel_set)
    fb2str = get_fb2str_dict()
    with open('../data/complex_data_webquestions.txt', 'b', encoding='utf-8') as f:
        for i, (question, triple, answer) in enumerate(zip(questions, triples, ans)):
            if i not in handel_set:
                a_list = answer.split(' <t>￨O ')
                t_list = triple.split('<t>')
                answers = []
                triple = []
                assert len(t_list) == len(a_list)
                for tt, aa in zip(t_list, a_list):
                    es = tt.split(' ')
                    aes = aa.split(' ')
                    for m, n in zip(es, aes):
                        if n.__contains__('A'):
                            answers.append(fb2str.get(m, m))
                    triple.append([fb2str.get(e, e) for e in es])
                answers = set(answers)
                print(question, triple, answers)
                question, question_e, answer, relation, triple = get_entity_and_relation(question, triple, answers)
                print(question, question_e, answer, relation, triple)
                try:
                    f.write(str(
                        i) + '\t' + question + '\t' + question_e + '\t' + relation + '\t' + answer + '\t' + triple + '\n')
                    f.flush()
                except Exception as e:
                    print('error:', e)
                    with open('../data/complex_error_line.txt', 'b', encoding='utf-8') as cf:
                        cf.write(str(i) + '\n')

def get_fb2w_dict(path='../data/fb2w.nt'):
    intermediate_fb2w = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            try:
                line_info = re.findall('<(.*?)>', line)
                intermediate_fb2w[line_info[0].split('/')[-1]] = line_info[2].split('/')[-1]
            except:
                print('error line:', line)
    return intermediate_fb2w


def prepare_simple_question(data_path='../data/fqFiltered.txt'):
    intermediate_data = []
    handel_set = []
    try:
        with open('../data/simple_data.txt', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if line.strip():
                    handel_set.append(int(line.strip().split('\t')[0]))
    except:
        handel_set = []

    fb2str = get_fb2str_dict()
    outfile = open('../data/simple_data.txt', 'b', encoding='utf-8')
    with open(data_path, 'r', encoding='utf-8') as outfile:
        for i, line in enumerate(f.readlines()):
            try:
                line_info = line.strip().replace('<http://rdf.freebase.com/ns/', '').replace('www.freebase.com/',
                                                                                             '').split('\t')
                question = line_info[3].lower()
                subject = line_info[0].replace('>', '').replace('/', '.')
                releation = line_info[1].replace('>', '').replace('/', '.')
                object = line_info[2].replace('>', '').replace('/', '.')
                triple = [[fb2str.get(subject, None), releation, fb2str.get(object, object)]]
                answers = [triple[0][2]]
                if triple[0][0]:
                    question, question_e, answer, relation, triple = get_entity_and_relation(question, triple, answers)
                    print(question, question_e, answer, relation, triple)
                    try:
                        outfile.write(str(
                            i) + '\t' + question + '\t' + question_e + '\t' + relation + '\t' + answer + '\t' + triple + '\n')
                        outfile.flush()
                    except:
                        with open('../data/simple_error_line.txt', 'b', encoding='utf-8') as sf:
                            sf.write(str(i) + '\n')
            except:
                print('error line:', line)
    outfile.close()


def main():
    # prepare_complex_question(question_path='../data/full.tgt', subgraph_path='../data/subgraph.txt',
    #                          ans_path='../data/train.src')

    get_final_preprocess_data(complex_data_path='./complex_data_webquestions.txt')


if __name__ == '__main__':
    main()

