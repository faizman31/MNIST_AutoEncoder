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
        indices = torch.randperm(x.shape[0],device=x.device)
        x = torch.index_select(x,dim=0,index=indices).split(config.batch_size,dim=0)
        y = torch.index_select(y,dim=0,index=indices).split(config.batch_size,dim=0)

        total_loss = 0 

        for i,(x_i,y_i) in enumerate(zip(x,y)):
            y_hat_i = self.model(x_i)
            loss_i= self.crit(y_hat_i,y_i.squeeze())

            self.optimizer.zero_grad()
            loss_i.backward()
            
            self.optimizer.step()

            if config.verbose >= 2:
                print('Train Iteration(%d/%d) : loss=%.4e' %(i+1,len(x),loss_i))

            total_loss = float(loss_i)
        
        return total_loss / len(x)

    def _validate(self,x,y):
        self.model.eval()

        with torch.no_grad():
            indices = torch.randperm(x.shape[0],device=x.device)

            x=torch.index_select(x,dim=0,index=indices).split(self.config.batch_size,dim=0)
            y=torch.index_select(y,dim=0,index=indices).split(self.config.batch_size,dim=0)

            total_loss = 0

            for i,(x_i,y_i) in enumerate(zip(x,y)):
                y_hat_i = self.model(x_i)
                loss = self.crit(y_hat_i,y_i.squeeze())
                

            







