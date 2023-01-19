import torch
import numpy as np

def load_mnist(is_train,flatten=True):
    from torchvision import datasets,transforms
    dataset = datasets.mnist(
        './data',
    train=is_train,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()])
    )

    x = dataset.data.float() / 255. # min/max Scaling
    y = dataset.target

    if flatten:
        x=x.reshape(x.shape[0],-1)

    return x,y

def split_data(x,y,train_ratio=.8):
    train_cnt = int(x.shape[0] * train_ratio)
    valid_cnt = x.shape[0] - train_cnt

    indices = torch.randperm(x.shape[0])
    x = torch.index_select(x,dim=0,index=indices).split([train_cnt,valid_cnt],dim=0)
    y = torch.index_select(y,dim=0,index=indices).split([train_cnt,valid_cnt],dim=0)

    return x,y


def get_encoder_hidden_size(n_layers):
    hidden_sizes=[]
    current_size=784 # MNIST 28*28 = 784
    step_size = int(784 / n_layers)
    for i in range(n_layers):
        hidden_sizes.append(current_size)
        current_size = current_size - step_size
    
    return hidden_sizes


