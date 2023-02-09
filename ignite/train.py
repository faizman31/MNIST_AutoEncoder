import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import *
from utils import get_encoder_hidden_size
from model import *

def define_argparser():
    
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn',required=True)
    p.add_argument('--gpu_id',type=int,default= 0 if torch.cuda.is_available() else -1)

    p.add_argument('--n_epochs',type=int,default=10)
    p.add_argument('--batch_size',type=int,default=256)
    p.add_argument('--train_ratio',type=float,default=.8)

    p.add_argument('--n_layers',type=int,default=5)
    p.add_argument('btl_size',type=int,default=2)
    p.add_argument('use_dropout',action='store_true')
    p.add_argument('dropout_p',type=float,default=.3)
    
    p.add_argument('verbose',type=int,default=1)

    config = p.parse_args()

    return config

def main(config):
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    train_loader,valid_loader,test_loader = get_loaders(config)

    print('Train : ',len(train_loader.dataset))
    print('Valid : ',len(valid_loader.dataset))
    print('Test : ',len(test_loader.dataset))

    model = AutoEncoder(
        input_size = 28**2,
        output_size = 28**2,
        hidden_sizes = get_encoder_hidden_size(config.n_layers),
        btl_size = config.btl_size
    ).to(device)

    optimizer = optim.Adam(model.parameters())
    crit = nn.MSELoss()

    trainer = Trainer(config)
    trainer.train(model,crit,optimizer,train_loader,valid_loader)

if __name__ == '__main__':
    config = define_argparser()
    main(config)

