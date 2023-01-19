import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import argparse

from model import AutoEncoder
from trainer import Trainer
from utils import *


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn',required=True)
    p.add_argument('--gpu_id',type=int,default=0 if torch.cuda.is_available() else -1)
    
    p.add_argument('--n_epochs',type=int,default=10)
    p.add_argument('--batch_size',type=int,default=256)
    p.add_argument('--train_ratio',type=float,default=.8)
    
    p.add_argument('--n_layers',type=int,default=5)
    p.add_argument('--btl_size',type=int,default=10)
    p.add_argument('--use_dropout',action='store_true')
    p.add_argument('--dropout_p',type=float,default=.3)

    p.add_argument('--verbose',type=int,default=1)

    config=p.parse_args()

    return config

def main(config):
    device = torch.device('cpu') if config.gpu_id <0 else torch.device('cuda:%d'%(config.gpu_id))

    x,y = load_mnist(is_train=True,flatten=True)
    x,y = split_data(x.to(device),y.to(device))
    
    print('Train : ',x[0].shape,y[0].shape)
    print('Valid : ',x[1].shape,y[1].shape)

    # AutoEncoder -> input_size == output_size
    input_size = int(x[0].shape[-1])
    output_size = int(x[0].shape[-1])

    model = AutoEncoder(
        input_size= input_size,
        output_size=output_size,
        hidden_sizes = get_encoder_hidden_size(config.n_layers),
        btl_size = config.btl_size,
        use_batch_norm = not config.use_dropout,
    ).to(device)

    optimizer = optim.Adam(model.parameters())
    crit = nn.NLLLoss()

    if config.verbose >=1:
        print(model)
        print(optimizer)
        print(crit)

    trainer = Trainer(model,optimizer,crit)

    trainer.train(
        train_data=(x[0],x[0]),
        valid_data=(x[1],x[1]),
        config=config)

    torch.save({
        'model' : trainer.model.state_dict(),
        'opt' : optimizer.state_dict(),
        'config':config
    },config.model_fn)


if __name__ == '__main__':
    config = define_argparser()
    main(config)


