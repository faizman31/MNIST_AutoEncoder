import torch
import numpy as np

def get_encoder_hidden_size(n_layers):
    hidden_sizes=[]
    current_size=784 # MNIST 28*28 = 784
    step_size = int(784 / n_layers)
    for i in range(n_layers):
        hidden_sizes.append(current_size)
        current_size = current_size - step_size
    
    return hidden_sizes


