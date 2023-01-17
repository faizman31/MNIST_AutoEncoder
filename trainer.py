import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *
import utils


class Trainer():

    def __init__(self,
                model,
                optimizer,
                crit,
                ):
        self.model=model
        self.optimizer=optimizer
        self.crit=crit

        super().__init__()

    def _train(self,x,y,config):
        self.model.train()
        # Shuffle before train
        indices = torch.randperm(x.shape[0])
        x = torch.index_select(x,dim=0,index=indices).split(config.batch_size,dim=0)
        y = torch.index_select(y,dim=0,index=indices).split(config.batch_size,dim=0)

        total_loss = 0 







