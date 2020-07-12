import math
import torch
import torch.nn as nn

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
        # since sigma is a diagonal matrix 1/sigma = simga^-1
        diff = y - mean
        # diff*diff.T*simga^-1
        quad = torch.bmm(diff.view(batch_size, inputs ,1), diff.view(batch_size, 1, inputs))*sigma.inverse()# trace argument
        # trace(diff*diff.T*sigma^-1)
        trace = torch.stack([torch.trace(q) for q in quad]) # compute trace
        loss = (torch.log(sigma_det) + trace).mean()
    return loss

class Swish(nn.Module):
    """
    Class that implements the 'swish' activation function:

        f(x) = x*sigmoid(beta*x)

    https://arxiv.org/pdf/1710.05941.pdf
    """
    def __init__(self, beta=1.0, trainable=True):
        super(Swish, self).__init__()
        if trainable:
            self.beta = Parameter(torch.tensor(0.1))
            self.beta.requiresGrad = True
        else:
            self.beta = beta


    def forward(self, input):
        return swish_function(input, beta=self.beta)


def swish_function(input, beta=1.0):
    """
    Swish activation function, f(x) = x*sigmoid(beta*x), (beta - scalar) 

    https://arxiv.org/pdf/1710.05941.pdf

    
    Args:
        input(torch.Tensor):
            input vector
        beta (float, torch.nn.parameter.Parameter):
            constant or trainable parameter
        trainable (bool):
            if 'True', 'beta' is a trainable parameter


    Returns:
        y (torch.Tensor):
            output of the swish activation function

    """
    y = input*torch.sigmoid(beta*input)
    return y


