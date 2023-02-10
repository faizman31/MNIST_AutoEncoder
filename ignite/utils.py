def get_encoder_hidden_size(n_layers):
    hidden_sizes=[]
    step_size = int((28*28) / n_layers-1)
    current_size = (28*28) - step_size # MNIST 28*28 = 784
    for i in range(n_layers-1):
        hidden_sizes.append(current_size)
        current_size = current_size - step_size
    
    return hidden_sizes


def get_grad_norm(parameters,norm_type=2):
    parameters = list(filter(lambda p : p.grad is not None,parameters))
    
    total_norm = 0

    try:
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm +=param_norm ** norm_type

    except Exception as e:
        print(e)

    return total_norm

def get_parameter_norm(parameters,norm_type=2):
    total_norm = 0

    try:
        for p in parameters:
            param_norm = p.data.norm(norm_type)
            total_norm += param_norm ** norm_type
    except Exception as e:
        print(e)

    return total_norm
    