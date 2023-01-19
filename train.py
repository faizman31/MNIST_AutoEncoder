import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import argparse

from model import AutoEncoder
from trainer import Trainer
from utils import load_mnist , get_encoder_hidden_size


def define_argparse():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn',required=True)
    p.add_arguemnt('--gpu_id',type=int,default=0 if torch.cuda.is_available() else -1)
    
    p.add_argument('--n_epochs',type=int,default=10)
    p.add_argument('--batch_size',type=int,default=256)
    p.add_argument('--train_ratio',type=float,default=.8)
    
    p.add_argument('--encoder_layers',type=int,default=5)
    p.add_arguemnt('--decoder_layers',type=int,default=5)
    p.add_argument('--use_dropout',action='store_true')
    p.add_argument('--dropout_p',type=float,default=.3)
    p.add_argument('--verbose',type=int,default=1)


    config=p.parse_args()

    return config

def main(config):
    x,y = load_mnist()
    
    train_cnt = int(x.shape[0] * config.train_ratio)
    valid_cnt = x.shape[0] - train_cnt