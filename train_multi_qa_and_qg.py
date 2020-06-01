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
from qg_qa.modules.optimizers.ScheduledOptim import ScheduledOptim
from qg_qa.question_generation.dataset import QG_and_QA_Dataset
from qg_qa.evaluating.bleu.bleu import Bleu
from qg_qa.evaluating.rouge.rouge import Rouge
import os

from qg_qa.relation_classification.models import Relation_classification
from qg_qa.question_generation.models import QG
from qg_qa.model import QG_And_QA
import sacrebleu

torch.cuda.set_device(0)


def cal_performance(pred, gold, smoothing=False):
    ''' Apply label smoothing if needed '''

    gold = gold.contiguous().view(-1)
    loss = cal_loss(pred, gold, smoothing)
    # print(pred, gold)
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
    qa_total_loss = 0
    qg_total_loss = 0

    n_word_total = 0
    n_word_correct = 0

    qa_n_word_total = 0
    qa_n_word_correct = 0

    # print(len(training_data))
    for batch in training_data:
        # print(len(batch))
        graph_spe, adj, question_spe, question_full, answer, graph_node_mask, graph_full, graph_spe_seq, question_spe_e, question_full_e, answer_seq, graph_node_mask_seq, graph_full_seq, relation, question2full_idx2idx = map(
            lambda x: x.to(device), batch)

        qg_gold = question_full[:, 1:]
        qa_gold = relation[:, 1:]

        optimizer.zero_grad()

        qg_output, qa_output, qa_qg_output = model(nodes=graph_spe, adj=adj, ans=answer, mask=graph_node_mask,
                                                   qidx2full=question2full_idx2idx, nidx2full=graph_full,
                                                   question=question_spe[:, :-1],
                                                   teacher_forcing_ratio=1, question_spe_e=question_spe_e,
                                                   rel=relation[:, :-1])

        qg_pred = qg_output.view(-1, qg_output.size(-1))
        qg_loss, n_correct = cal_performance(qg_pred, qg_gold, smoothing=smoothing)

        qa_pred = qa_output.view(-1, qa_output.size(-1))
        qa_gold = qa_gold.contiguous().view(-1)
        qa_loss = F.cross_entropy(qa_pred, qa_gold, ignore_index=0, reduction='sum')

        qa_qg_pred = qa_qg_output.view(-1, qa_qg_output.size(-1))
        qa_qg_loss = F.cross_entropy(qa_qg_pred, qa_gold, ignore_index=0, reduction='sum')

        loss = qg_loss + qa_loss + qa_qg_loss
        loss.backward()

        # update parameters
        optimizer.step()

        # note keeping
        total_loss += loss.item()
        qa_total_loss += qa_loss.item()
        qg_total_loss += qg_loss.item()

        non_pad_mask = qa_gold.ne(0)
        qa_pred = qa_pred.max(1)[1]
        qa_n_correct = qa_pred.eq(qa_gold)
        qa_n_correct = qa_n_correct.masked_select(non_pad_mask).sum().item()
        qa_n_word = non_pad_mask.sum().item()
        qa_n_word_total += qa_n_word
        qa_n_word_correct += qa_n_correct

        non_pad_mask = qg_gold.ne(0)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct

    qa_loss_per_word = qa_total_loss / qa_n_word_total
    qa_accuracy = qa_n_word_correct / qa_n_word_total
    loss_per_word = qg_total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    return loss_per_word, accuracy, qa_loss_per_word, qa_accuracy


def eval_epoch(model, validation_data, device):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    qa_total_loss = 0
    qg_total_loss = 0

    n_word_total = 0
    n_word_correct = 0

    qa_n_word_total = 0
    qa_n_word_correct = 0

    with torch.no_grad():
        for batch in validation_data:
            # prepare data
            graph_spe, adj, question_spe, question_full, answer, graph_node_mask, graph_full, graph_spe_seq, question_spe_e, question_full_e, answer_seq, graph_node_mask_seq, graph_full_seq, relation, question2full_idx2idx = map(
                lambda x: x.to(device), batch)

            qg_gold = question_full[:, 1:]
            qa_gold = relation[:, 1:]

            qg_output, qa_output, qa_qg_output = model(nodes=graph_spe, adj=adj, ans=answer, mask=graph_node_mask,
                                                       qidx2full=question2full_idx2idx, nidx2full=graph_full,
                                                       question=question_spe[:, :-1],
                                                       teacher_forcing_ratio=0, question_spe_e=question_spe_e,
                                                       rel=relation[:, :-1])

            qg_pred = qg_output.view(-1, qg_output.size(-1))
            qg_loss, n_correct = cal_performance(qg_pred, qg_gold, smoothing=False)

            qa_pred = qa_output.view(-1, qa_output.size(-1))
            qa_gold = qa_gold.contiguous().view(-1)
            qa_loss = F.cross_entropy(qa_pred, qa_gold, ignore_index=0, reduction='sum')

            qa_qg_pred = qa_qg_output.view(-1, qa_qg_output.size(-1))
            qa_qg_loss = F.cross_entropy(qa_qg_pred, qa_gold, ignore_index=0, reduction='sum')

            loss = qg_loss + qa_loss + qa_qg_loss

            # note keeping
            total_loss += loss.item()
            qa_total_loss += qa_loss.item()
            qg_total_loss += qg_loss.item()

            non_pad_mask = qa_gold.ne(0)
            qa_pred = qa_pred.max(1)[1]
            qa_n_correct = qa_pred.eq(qa_gold)
            qa_n_correct = qa_n_correct.masked_select(non_pad_mask).sum().item()
            qa_n_word = non_pad_mask.sum().item()
            qa_n_word_total += qa_n_word
            qa_n_word_correct += qa_n_correct

            non_pad_mask = qg_gold.ne(0)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

    qa_loss_per_word = qa_total_loss / qa_n_word_total
    qa_accuracy = qa_n_word_correct / qa_n_word_total
    loss_per_word = qg_total_loss / n_word_total
    accuracy = n_word_correct / n_word_total
    return loss_per_word, accuracy, qa_loss_per_word, qa_accuracy


def train(model, training_data, validation_data, optimizer, device, opt):
    ''' Start training '''

    log_train_file = None
    log_valid_file = None

    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'

        print('[Info] Training performance will be written to file: {} and {}'.format(
            log_train_file, log_valid_file))

        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')

    # show_performance(model, validation_data, device)

    valid_accus = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')

        start = time.time()
        train_loss, train_accu, qa_loss, qa_accu = train_epoch(
            model, training_data, optimizer, device, smoothing=opt.label_smoothing)
        print(
            '  - (Training)   ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %,    ppl: {qa_ppl: 8.5f}, accuracy: {qa_accu:3.3f} %,' \
            'elapse: {elapse:3.3f} min'.format(
                ppl=math.exp(min(train_loss, 100)), accu=100 * train_accu, qa_ppl=math.exp(min(qa_loss, 100)),
                qa_accu=100 * qa_accu,
                elapse=(time.time() - start) / 60))

        start = time.time()
        valid_loss, valid_accu, qa_loss, qa_accu = eval_epoch(model, validation_data, device)
        print(
            '  - (Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %,     ppl: {qa_ppl: 8.5f}, accuracy: {qa_accu:3.3f} %,' \
            'elapse: {elapse:3.3f} min'.format(
                ppl=math.exp(min(valid_loss, 100)), accu=100 * valid_accu, qa_ppl=math.exp(min(qa_loss, 100)),
                qa_accu=100 * qa_accu,
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

        # show_performance(model, validation_data, device)
        if log_train_file and log_valid_file:
            with open(log_train_file, 'b') as log_tf, open(log_valid_file, 'b') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=train_loss,
                    ppl=math.exp(min(train_loss, 100)), accu=100 * train_accu))
                log_vf.write('{epoch},{loss: 8.5f},{ppl: 8.5f},{accu:3.3f}\n'.format(
                    epoch=epoch_i, loss=valid_loss,
                    ppl=math.exp(min(valid_loss, 100)), accu=100 * valid_accu))


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

            qg_output, qa_output, qa_qg_output = model(nodes=graph_spe, adj=adj, ans=answer, mask=graph_node_mask,
                                                       qidx2full=question2full_idx2idx, nidx2full=graph_full,
                                                       question=question_spe[:, :-1],
                                                       teacher_forcing_ratio=0, question_spe_e=question_spe_e,
                                                       rel=relation[:, :-1])

            pred = qg_output.max(2)[1]
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
    parser.add_argument('-save_model', default='./saved_models/QG_AND_QA@V1',
                        help='训练后模型保存路径')
    parser.add_argument('-save_mode', default='best', help='训练后模型保存路径')
    parser.add_argument('-use_copy', default=True, help='是否使用copy机制')
    parser.add_argument('-encoder_type', default='GNN', help='encoder类型')
    parser.add_argument('-max_len', default=24, help='生成句子最大长度')
    parser.add_argument('-log', default=None, help='训练日志保存路径')
    parser.add_argument('-n_warmup_steps', type=int, default=4000)
    parser.add_argument('-lr', type=float, default=0.5, help='Initial learning rate.')
    parser.add_argument('-weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-label_smoothing', default=False)
    parser.add_argument('-sparse', action='store_true', default=False, help='GAT with sparse version or not.')

    opt = parser.parse_args()

    if opt.save_model and not os.path.exists(os.path.dirname(opt.save_model)):
        os.mkdir(os.path.dirname(opt.save_model))
    if opt.log and not os.path.exists(os.path.dirname(opt.log)):
        os.mkdir(os.path.dirname(opt.log))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ========= Loading Dataset =========#
    data = torch.load(opt.data)
    training_data, validation_data = prepare_dataloaders(data, opt)

    opt.src_vocab_size = training_data.dataset.src_vocab_size
    opt.tgt_vocab_size = training_data.dataset.tgt_vocab_size

    print(opt.tgt_vocab_size)
    # print(len(training_data), len(validation_data))

    # qg_model = QG(node_vocab_size=training_data.dataset.src_vocab_size,
    #              q_vocab_size=training_data.dataset.tgt_vocab_size,
    #              full_vocab_size=training_data.dataset.full_vocab_size, max_len=opt.max_len, e_emb_dim=256,
    #              q_emb_dim=512,  hidden_size=opt.d_model, sos_id=2, nheads=3, alpha=0.2,
    #              dropout=0.1, use_copy=opt.use_copy,encoder_type=opt.encoder_type)
    # qa_model = Relation_classification(source_vocab_size=training_data.dataset.tgt_vocab_size, hidden_size=256,
    #                                 target_vocab_size=training_data.dataset.rel_vocab_size).to(device)
    #
    # qg_model = qg_model.to(device)
    # qa_model = qa_model.to(device)
    model = QG_And_QA(node_vocab_size=training_data.dataset.src_vocab_size,
                      q_vocab_size=training_data.dataset.tgt_vocab_size,
                      full_vocab_size=training_data.dataset.full_vocab_size, max_len=opt.max_len, e_emb_dim=256,
                      q_emb_dim=512, hidden_size=opt.d_model, sos_id=2, nheads=3, alpha=0.2,
                      dropout=0.1, use_copy=opt.use_copy, encoder_type=opt.encoder_type,
                      source_vocab_size=training_data.dataset.tgt_vocab_size, qa_hidden_size=256,
                      target_vocab_size=training_data.dataset.rel_vocab_size,full2spe=training_data.dataset._full2question_idx2idx).to(device)
    model = model.to(device)
    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        opt.d_model, opt.n_warmup_steps)

    # optimizer = optim.Adam(model.parameters(),
    #     #                        lr=opt.lr,
    #     #                        weight_decay=opt.weight_decay)

    try:
        model.load_state_dict(torch.load(opt.save_model + '_best.chkpt')['model'])
        print('loading the old model')
    except:
        model.qg.load_state_dict((torch.load('./saved_models/QG@V2_best.chkpt')['model']))
        model.qa.load_state_dict((torch.load('./saved_models/QA@V2_best.chkpt')['model']))
        print('loading sub_model')

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
        # collate_fn=paired_collate_fn,
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
        # collate_fn=paired_collate_fn,
        shuffle=False)
    return train_loader, valid_loader


if __name__ == '__main__':
    main()
