import torch
import numpy as np

def load_mnist(is_train,flatten=True):
    from torchvision import datasets,transforms
    dataset = datasets.mnist('../data',
    train=is_train,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()])
    )

    x = dataset.data.float() / 255. # min/max Scaling
    y = dataset.target

    if flatten:
        x=x.reshape(x.shape[0],-1)

    return x,y


def get_encoder_hidden_size(n_layers):
    hidden_sizes=[]
    current_size=784 # MNIST 28*28 = 784
    step_size = int(784 / n_layers)
    for i in range(n_layers):
        hidden_sizes.append(current_size)
        current_size = current_size - step_size
    
    return hidden_sizes


