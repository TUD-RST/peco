import torch
from torch.nn.parameter import Parameter

def NLLloss(y, mean, var):
    """
    Negative log-likelihood loss function of a Gaussian.

    see eq. (1) in https://arxiv.org/abs/1612.01474

    Args:
        y (torch.Tensor):
            target output vector of shape (batch_size, outputs)
        mean (torch.Tensor):
            mean vector of shape (batch_size, inputs)
        var (torch.Tensor):
            variance vector of shape (batch_size, inputs)

    Returns:
        loss (torch.tensor):
            loss

    """
    batch_size, inputs = mean.shape
    if inputs == 1:
        loss = (torch.log(var) + ((y - mean).pow(2))/var).mean()
    else:
        sigma = torch.stack([torch.diag(v) for v in var]) # calc sigmas^-1 shape = [B, inputs, inputs]
        sigma_det = torch.stack([torch.det(s) for s in sigma])
        diff = y - mean
        quad = torch.bmm(diff.view(batch_size, inputs ,1), diff.view(batch_size, 1, inputs))*sigma.inverse() # trace argument
        exp = torch.stack([torch.trace(q) for q in quad]) # compute trace
        loss = (torch.log(sigma_det) + exp).mean()
    return loss

def swish(x, beta=0.01, trainable=True):
    """
    Swish activation function, f(x) = x*sigmoid(beta*x), (beta - scalar) 

    https://arxiv.org/pdf/1710.05941.pdf

    
    Args:
        x (torch.Tensor):
            input vector
        beta (float): 
            constant or trainable parameter
        trainable (bool):
            if 'True', 'beta' is a trainable parameter


    Returns:
        y (torch.Tensor):
            output of the swish activation function

    """
    if trainable:
        beta = Parameter(torch.Tensor(1, 1))
    y = x*torch.sigmoid(beta*x)
    return y