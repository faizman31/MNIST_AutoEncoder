def get_encoder_hidden_size(n_layers):
    hidden_sizes=[]
    step_size = int((28*28) / n_layers-1)
    current_size = (28*28) - step_size # MNIST 28*28 = 784
    for i in range(n_layers-1):
        hidden_sizes.append(current_size)
        current_size = current_size - step_size
    
    return hidden_sizes