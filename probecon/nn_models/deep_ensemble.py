import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class MLP(nn.Module):
    """ Multilayer perceptron (MLP) with tanh/sigmoid activation functions implemented in PyTorch for regression tasks.

    Attributes:
        inputs (int): inputs of the network
        outputs (int): outputs of the network
        hidden_layers (list): layer structure of MLP: [5, 5] (2 hidden layer with 5 neurons)
        activation (string): activation function used ('relu', 'tanh' or 'sigmoid')

    """

    def __init__(self, inputs=1, outputs=1, hidden_layers=[100], activation='relu'):
        super(MLP, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.hidden_layers = hidden_layers
        self.nLayers = len(hidden_layers)
        self.net_structure = [inputs, *hidden_layers, outputs]

        if activation == 'relu':
            self.act = torch.relu
        elif activation == 'tanh':
            self.act = torch.tanh
        elif activation == 'sigmoid':
            self.act = torch.sigmoid
        else:
            assert ('Use "relu","tanh" or "sigmoid" as activation.')
        # create linear layers y = Wx + b

        for i in range(self.nLayers + 1):
            setattr(self, 'layer_' + str(i), nn.Linear(self.net_structure[i], self.net_structure[i + 1]))

    def forward(self, x):
        # connect layers
        for i in range(self.nLayers):
            layer = getattr(self, 'layer_' + str(i))
            x = self.act(layer(x))
        layer = getattr(self, 'layer_' + str(self.nLayers))
        x = layer(x)
        return x

class GaussianMLP(MLP):
    """ Gaussian MLP which outputs are mean and variance.

    Attributes:
        inputs (int): number of inputs
        outputs (int): number of outputs
        hidden_layers (list of ints): hidden layer sizes

    """

    def __init__(self, inputs=1, outputs=1, hidden_layers=[100], activation='relu', std_max=None, correlated=False):
        super(GaussianMLP, self).__init__(inputs=inputs, outputs=2*outputs, hidden_layers=hidden_layers, activation=activation)
        self.inputs = inputs
        self.outputs = outputs
        self.std_max = std_max # torch tensor of shape (outputs,)
    def forward(self, x):
        # connect layers
        for i in range(self.nLayers):
            layer = getattr(self, 'layer_'+str(i))
            x = self.act(layer(x))
        layer = getattr(self, 'layer_' + str(self.nLayers))
        x = layer(x)
        mean, std = torch.split(x, self.outputs, dim=1)
        if self.std_max is not None:
            std = (self.std_max*torch.sigmoid(std))
        else:
            std = F.softplus(std)
        variance = std.pow(2) + 1e-6
        return mean, variance


class DeepEnsemble(nn.Module):
    """ Deep Ensemble (Gaussian mixture MLP) which outputs are mean and variance.

    Attributes:
        models (int): number of models
        inputs (int): number of inputs
        outputs (int): number of outputs
        hidden_layers (list of ints): hidden layer sizes

    """

    def __init__(self, num_models=5, inputs=1, outputs=1, hidden_layers=[100], activation='relu', std_max=None, adversarial=False, eps=0.001):
        super(DeepEnsemble, self).__init__()
        self.num_models = num_models
        self.inputs = inputs
        self.outputs = outputs
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.adversarial = adversarial
        self.eps = eps # epsilon in adversarial training
        for i in range(self.num_models):
            model = GaussianMLP(inputs=self.inputs,
                                outputs=self.outputs,
                                hidden_layers=self.hidden_layers,
                                activation=self.activation,
                                std_max=std_max)
            setattr(self, 'model_' + str(i), model)

    def forward(self, x):
        # connect layers
        means = []
        variances = []
        for i in range(self.num_models):
            model = getattr(self, 'model_' + str(i))
            mean, var = model(x)
            means.append(mean)
            variances.append(var)
        means = torch.stack(means)
        mean = means.mean(dim=0)
        variances = torch.stack(variances)
        sigma2 = variances.mean(dim=0) # mean of the variances
        sigma2_mod = means.pow(2).mean(dim=0) - mean.pow(2)  # variance of means
        variance = sigma2 + sigma2_mod
        return mean, variance, sigma2_mod

    def training(self, x, y, batchsize, epochs, loss='nll', lr=1e-3, weight_decay=1e-5):
        """ Training process. """

        # initialize optimizers
        for i in range(self.num_models):
            model = getattr(self, 'model_' + str(i))
            model.optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(epochs):
            # shuffle data
            shuffle_idx = torch.randperm(x.__len__())
            for xb, yb in zip(x[shuffle_idx].split(batchsize), y[shuffle_idx].split(batchsize)):
                losses = self._train_ensemble(xb, yb, loss)
            if epoch == 0:
                print('inital loss: ', np.mean(np.array(losses)))
        print('final loss: ', np.mean(np.array(losses)))



    def _train_ensemble(self, x, y, loss):
        """ One training step of gradient descent on the whole deep ensemble """

        for i in range(self.num_models):
            model = getattr(self, 'model_' + str(i))
            self._train_submodel(x, y, model, loss)


    def _train_submodel(self, x, y, model, loss):
        """ One training step of gradient descent on an individual submodel of the deep ensemble """

        # todo: check if requires_grad = True is necessary
        x.requires_grad = True
        model.optimizer.zero_grad()
        mean, var = model(x)
        if loss == 'nll':
            loss = NLLloss(y, mean, var)
        if loss == 'mse':
            loss = nn.MSELoss(y, mean)
        loss.backward()
        if self.adversarial:  # adversarial training
            x_prime = x + self.eps * torch.sign(x.grad)  # create adversarial examples
            mean, var = model(x)
            mean_prime, var_prime = model(x_prime)
            loss = NLLloss(y, mean, var) + NLLloss(y, mean_prime, var_prime)
            loss.backward()
        model.optimizer.step()
        x.requires_grad = False
        return loss.item()


def NLLloss(y, mean, var):
    """ Negative log-likelihood loss function of a Gaussian. """
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

