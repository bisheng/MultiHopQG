# -*- coding: utf-8 -*-
"""
Created by fx at 2019-10-20 15:01:38
==============================
dcqg_train
"""

import argparse
import math
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from modules.optimizers.ScheduledOptim import ScheduledOptim
from data.dataset import QG_and_QA_Dataset
from evaluating.bleu.bleu import Bleu
from evaluating.rouge.rouge import Rouge
import os

from models.question_generaion import QG

import sacrebleu  # another compute bleu score method

torch.cuda.set_device(0)


def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

    gold = gold.contiguous().view(-1)
    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]

    non_pad_mask = gold.ne(0)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def cal_loss(pred, gold, smoothing):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(0)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=0, reduction='sum')

    return loss


def train_epoch(model, training_data, optimizer, device, smoothing):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    for batch in training_data:
        graph_spe, adj, question_spe, question_full, answer, graph_node_mask, graph_full, graph_spe_seq, question_spe_e, question_full_e, answer_seq, graph_node_mask_seq, graph_full_seq, relation, question2full_idx2idx = map(
            lambda x: x.to(device), batch)

        gold = question_full[:, 1:]

        # forward
        optimizer.zero_grad()
        pred, _ = model(nodes=graph_spe, adj=adj, ans=answer, mask=graph_node_mask,
                        qidx2full=question2full_idx2idx, nidx2full=graph_full, question=question_spe[:, :-1],
                        teacher_forcing_ratio=1)

        pred = pred.view(-1, pred.size(-1))

        # backward
        loss, n_correct = cal_performance(pred, gold, smoothing=smoothing)
        loss.backward()

        # update parameters
        optimizer.step()

        # note keeping
        total_loss += loss.item()
        non_pad_mask = gold.ne(0)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct

    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    return loss_per_word, accuracy


def eval_epoch(model, validation_data, device):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    with torch.no_grad():
        for batch in validation_data:
            # prepare data
            graph_spe, adj, question_spe, question_full, answer, graph_node_mask, graph_full, graph_spe_seq, question_spe_e, question_full_e, answer_seq, graph_node_mask_seq, graph_full_seq, relation, question2full_idx2idx = map(
                lambda x: x.to(device), batch)

            gold = question_full[:, 1:]

            # forward
            pred, _ = model(nodes=graph_spe, adj=adj, ans=answer, mask=graph_node_mask,
                            qidx2full=question2full_idx2idx, nidx2full=graph_full, question=question_spe[:, :-1],
                            teacher_forcing_ratio=0)
            pred = pred.view(-1, pred.size(-1))

            loss, n_correct = cal_performance(pred, gold, smoothing=False)

            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold.ne(0)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

    loss_per_word = total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    return loss_per_word, accuracy


def train(model, training_data, validation_data, optimizer, device, opt):
    ''' Start training '''

    valid_accus = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu = train_epoch(
            model, training_data, optimizer, device, smoothing=opt.label_smoothing)
        print('  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, ' \
              'elapse: {elapse:3.3f} min'.format(
            ppl=math.exp(min(train_loss, 100)), accu=100 * train_accu,
            elapse=(time.time() - start) / 60))

        start = time.time()
        valid_loss, valid_accu = eval_epoch(model, validation_data, device)
        print('  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, ' \
              'elapse: {elapse:3.3f} min'.format(
            ppl=math.exp(min(valid_loss, 100)), accu=100 * valid_accu,
            elapse=(time.time() - start) / 60))

        valid_accus += [valid_accu]

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i}

        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100 * valid_accu)
                torch.save(checkpoint, model_name)
            elif opt.save_mode == 'best':
                model_name = opt.save_model + '_best.chkpt'
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.')


def show_performance(model, data, device):
    model.eval()
    gts = {}
    res = {}
    count = 0
    with torch.no_grad():
        for batch in data:
            # prepare data
            graph_spe, adj, question_spe, question_full, answer, graph_node_mask, graph_full, graph_spe_seq, question_spe_e, question_full_e, answer_seq, graph_node_mask_seq, graph_full_seq, relation, question2full_idx2idx = map(
                lambda x: x.to(device), batch)

            gold = question_full[:, 1:]

            pred, _ = model(nodes=graph_spe, adj=adj, ans=answer, mask=graph_node_mask,
                            qidx2full=question2full_idx2idx, nidx2full=graph_full, question=question_spe[:, :-1],
                            teacher_forcing_ratio=0)

            pred = pred.max(2)[1]
            for p, g in zip(pred, gold):
                pred_line = []
                gold_line = []
                for idx in p:
                    if idx.item() != 3:
                        pred_line.append(data.dataset.full_idx2word[idx.item()])
                    else:
                        break
                for idx in g:
                    if idx.item() != 3:
                        gold_line.append(data.dataset.full_idx2word[idx.item()])
                    else:
                        break
                if count < 200:
                    print('gold:', ' '.join(gold_line))
                    print('pred:', ' '.join(pred_line))
                count += 1
                gts[count] = [' '.join(gold_line)]
                res[count] = [' '.join(pred_line)]
        avg_bleu = 0
        for key in gts.keys():
            avg_bleu += sacrebleu.corpus_bleu(res[key], [gts[key]]).score
        print('Single Bleu:', avg_bleu / len(gts.keys()))
        b = Bleu()
        print('BLEU: ', b.compute_score(gts, res)[0])
        r = Rouge()
        print('ROUGE_L: ', r.compute_rouge_l_score(gts, res)[0])
        for i in range(4):
            print('ROUGE_%s: ' % (i + 1), r.compute_rouge_n_score(gts, res, i + 1)[0])


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', default='./data/complex_data.pth')

    parser.add_argument('-epoch', type=int, default=100)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-save_model', default='./saved_models/QG@V1',
                        help='训练后模型保存路径')
    parser.add_argument('-save_mode', default='best', help='save the best or save all')
    parser.add_argument('-use_copy', default=True, help='是否使用copy机制')
    parser.add_argument('-encoder_type', default='GNN', help='encoder类型')
    parser.add_argument('-max_len', default=24, help='生成句子最大长度')
    parser.add_argument('-n_warmup_steps', type=int, default=4000)
    parser.add_argument('-lr', type=float, default=0.005, help='Initial learning rate.')
    parser.add_argument('-weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-label_smoothing', default=True)
    parser.add_argument('-sparse', action='store_true', default=False, help='GAT with sparse version or not.')

    opt = parser.parse_args()

    if opt.save_model and not os.path.exists(os.path.dirname(opt.save_model)):
        os.mkdir(os.path.dirname(opt.save_model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ========= Loading Dataset =========#
    data = torch.load(opt.data)
    training_data, validation_data = prepare_dataloaders(data, opt)

    opt.src_vocab_size = training_data.dataset.src_vocab_size
    opt.tgt_vocab_size = training_data.dataset.tgt_vocab_size

    print(opt.tgt_vocab_size)

    model = QG(node_vocab_size=training_data.dataset.src_vocab_size,
               q_vocab_size=training_data.dataset.tgt_vocab_size,
               full_vocab_size=training_data.dataset.full_vocab_size, max_len=opt.max_len, e_emb_dim=256,
               q_emb_dim=512, hidden_size=opt.d_model, sos_id=2, nheads=3, alpha=0.2,
               dropout=0.1, use_copy=opt.use_copy, encoder_type=opt.encoder_type)

    model = model.to(device)
    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        opt.d_model, opt.n_warmup_steps)

    try:
        model.load_state_dict(torch.load(opt.save_model + '_best.chkpt')['model'])
        print('loading the old model')
    except:
        pass

    train(model, training_data, validation_data, optimizer, device, opt)
    # show_performance(model, validation_data, device)


def prepare_dataloaders(data, opt):
    # ========= Preparing DataLoader =========#
    train_loader = torch.utils.data.DataLoader(
        QG_and_QA_Dataset(
            graph_word2idx=data['dict']['graph_node'],
            question_word2idx=data['dict']['question_words'],
            rel_word2idx=data['dict']['relation'],
            full_word2idx=data['dict']['full'],
            graph_spe=data['train']['graph_spe'],
            graph_full=data['train']['graph_full'],
            graph_spe_seq=data['train']['graph_spe_seq'],
            graph_full_seq=data['train']['graph_full_seq'],
            adj=data['train']['adj'],
            question_spe=data['train']['question_spe'],
            question_full=data['train']['question_full'],
            question_spe_e=data['train']['question_spe_e'],
            question_full_e=data['train']['question_full_e'],
            answer=data['train']['answer'],
            answer_seq=data['train']['answer_seq'],
            graph_node_mask_seq=data['train']['graph_node_mask_seq'],
            relation=data['train']['relation'],
            graph_node_mask=data['train']['graph_node_mask']),
        num_workers=0,
        batch_size=opt.batch_size,
        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
        QG_and_QA_Dataset(
            graph_word2idx=data['dict']['graph_node'],
            question_word2idx=data['dict']['question_words'],
            full_word2idx=data['dict']['full'],
            rel_word2idx=data['dict']['relation'],
            graph_spe=data['test']['graph_spe'],
            graph_full=data['test']['graph_full'],
            graph_spe_seq=data['test']['graph_spe_seq'],
            graph_full_seq=data['test']['graph_full_seq'],
            adj=data['test']['adj'],
            question_spe=data['test']['question_spe'],
            question_full=data['test']['question_full'],
            question_spe_e=data['test']['question_spe_e'],
            question_full_e=data['test']['question_full_e'],
            answer=data['test']['answer'],
            answer_seq=data['test']['answer_seq'],
            relation=data['test']['relation'],
            graph_node_mask_seq=data['test']['graph_node_mask_seq'],
            graph_node_mask=data['test']['graph_node_mask']),
        num_workers=0,
        batch_size=opt.batch_size,
        shuffle=False)
    return train_loader, valid_loader


if __name__ == '__main__':
    main()
