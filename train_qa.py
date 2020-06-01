import argparse
import math
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from modules.optimizers.ScheduledOptim import ScheduledOptim
from data.dataset import QG_and_QA_Dataset

import os

from models.relation_classification import Relation_classification

torch.cuda.set_device(0)

def run_epoch(model, data, device, max_len=None, optimizer=None, TRAIN=True):
    if TRAIN:
        model.train()
        total_loss = 0
        n_word_total = 0
        n_word_correct = 0
        for batch in data:
            _, _, _, _, _, _, _, _, question_spe_e, _, _, _, _, relation, _ = map(lambda x: x.to(device), batch)
            gold = relation[:, 1:]
            optimizer.zero_grad()

            pred, hidden = model(question_spe_e, relation[:, :-1])
            # print(pred.size(), gold.size())
            pred = pred.view(-1, pred.size(-1))
            gold = gold.contiguous().view(-1)

            loss = F.cross_entropy(pred, gold, ignore_index=0, reduction='sum')
            loss.backward()
            # update parameters
            optimizer.step()
            # note keeping
            non_pad_mask = gold.ne(0)
            pred = pred.max(1)[1]
            n_correct = pred.eq(gold)
            n_correct = n_correct.masked_select(non_pad_mask).sum().item()
            total_loss += loss.item()
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct

        loss_per_word = total_loss / n_word_total
        accuracy = n_word_correct / n_word_total
        return loss_per_word, accuracy
    else:
        model.eval()
        total_loss = 0
        n_word_total = 0
        n_word_correct = 0
        with torch.no_grad():
            for batch in data:
                _, _, _, _, _, _, _, _, question_spe_e, _, _, _, _, relation, _ = map(lambda x: x.to(device), batch)
                gold = relation[:, 1:]
                batch_size = question_spe_e.size(0)
                decoder_input = torch.LongTensor([2] * batch_size).view(batch_size, 1).to(device)
                _, hidden = model.encoder(question_spe_e)
                pred = torch.zeros(batch_size, max_len, model.decoder.output_size).to(device)
                for di in range(max_len):
                    decoder_output, hidden = model.decoder(decoder_input, hidden)
                    step_output = decoder_output.squeeze(1)
                    decoder_input = step_output.topk(1)[1]
                    pred[:, di, :] = step_output

                # print(pred.size(), gold.size())
                pred = pred.view(-1, pred.size(-1))
                gold = gold.contiguous().view(-1)

                loss = F.cross_entropy(pred, gold, ignore_index=0, reduction='sum')

                non_pad_mask = gold.ne(0)
                pred = pred.max(1)[1]
                n_correct = pred.eq(gold)
                n_correct = n_correct.masked_select(non_pad_mask).sum().item()
                total_loss += loss.item()
                n_word = non_pad_mask.sum().item()
                n_word_total += n_word
                n_word_correct += n_correct

        loss_per_word = total_loss / n_word_total
        accuracy = n_word_correct / n_word_total
        return loss_per_word, accuracy


def train(model, train_data, test_data, device, optimizer, opt):
    best_val_acc = 0
    for iter in range(0, opt.epoch):
        print('Epoch:', iter)
        loss_per_word, accuracy = run_epoch(model, train_data, device, optimizer=optimizer, TRAIN=True)
        print('training loss:{loss:.5f}, acc:{acc:3.3f}'.format(loss=loss_per_word, acc=100 * accuracy))
        val_loss_per_word, val_accuracy = run_epoch(model, test_data, device, max_len=opt.max_len, TRAIN=False)
        print('Val loss:{loss:.5f}, acc:{acc:3.3f}'.format(loss=val_loss_per_word, acc=100 * val_accuracy))
        if val_accuracy > best_val_acc:
            best_val_acc=val_accuracy
            model_state_dict = model.state_dict()
            checkpoint = {
                'model': model_state_dict,
                'settings': opt,
                'epoch': iter}
            model_name = opt.save_model + '_best.chkpt'
            torch.save(checkpoint, model_name)
            print('    - [Info] The checkpoint file has been updated.')


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', default='./data/complex_data.pth')

    parser.add_argument('-epoch', type=int, default=100)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-save_model', default='./saved_models/QA@V2',
                        help='训练后模型保存路径')
    parser.add_argument('-save_mode', default='best', help='训练后模型保存路径')
    parser.add_argument('-max_len', default=4, help='生成关系最多个数')
    parser.add_argument('-n_warmup_steps', type=int, default=4000)
    parser.add_argument('-lr', type=float, default=0.5, help='Initial learning rate.')
    parser.add_argument('-weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('-d_model', type=int, default=256)
    parser.add_argument('-label_smoothing', default=False)
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Relation_classification(source_vocab_size=training_data.dataset.tgt_vocab_size, hidden_size=256,
                                    target_vocab_size=training_data.dataset.rel_vocab_size).to(device)
    # optimizer = optim.SGD(model.parameters(), lr=0.5)
    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        opt.d_model, opt.n_warmup_steps)
    try:
        # model.load_state_dict(torch.load('./saved_models/QA@V1' + '_best.chkpt')['model'])
        model.load_state_dict(torch.load(opt.save_model + '_best.chkpt')['model'])
        print('loading the old model')
    except:
        pass

    train(model, training_data, validation_data, device, optimizer, opt)


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
