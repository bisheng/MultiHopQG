"""
# config.py created by bisheng at 2020/5/31 22:52.
"""
import argparse
import torch
import datetime
device = "cuda" if torch.cuda.is_available() else "cpu"
# 模型保存为 current_model.ckpt
# e.g., 20200525123421_rcnn.ckpt
current = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

def train_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', type=str)
    parser.add_argument('-train', type=str)
    parser.add_argument('-n_epochs', type=int, default=50)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-print_interval', type=int, default=10)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-src_max_len', type=int, default=21)
    parser.add_argument('-tgt_max_len', type=int, default=55)
    parser.add_argument('-checkpoint', type=int, default=0)
    cfg = parser.parse_args()
    cfg.device = device
    return cfg
