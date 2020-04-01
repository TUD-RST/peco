import torch
import torch.nn as nn
import numpy as np

from probecon.nn_models.mlp import GaussianMLP
from probecon.helpers.nn_helpers import NLLloss

class DeepEnsemble(nn.Module):
    """
    Class that implements a deep ensemble neural network model.

    https://arxiv.org/abs/1612.01474

    """

    def __init__(self,
                 num_models=5,
                 inputs=1,
                 outputs=1,
                 hidden_layers=[100],
                 activation='relu',
                 sparse=False,
                 std_max=None,
                 adversarial=False,
                 eps=0.001):
        """

        Args:
            num_models (int):
                number of models in the ensemble
            inputs (int):
                number of inputs
            outputs (int):
                number of outputs
            hidden_layers (list):
                layer structure of MLP: [5, 5] (2 hidden layer with 5 neurons)
            activation (str)
                activation function used ('relu', 'tanh', 'sigmoid' or 'swish')
            std_max (torch.Tensor):
                maximum standard deviation with shape (outputs,)
            sparse (bool):
                if 'True', for each output dimension a single MLP is used
            std_max (torch.Tensor):
                maximum values of the standard deviation
            adversarial (bool):
                if 'True', adverserial training is performed
            eps (float):
                parameter for adverserial training
        """
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
                                sparse=sparse,
                                std_max=std_max)
            setattr(self, 'model_' + str(i), model)

    def forward(self, x):
        """
        Forward pass of the deep ensemble.

        Args:
            x (torch.Tensor):
                input tensor

        Returns:
            mean (torch.Tensor):
                mean of the model for the given input 'x'
            variance (torch.Tensor):
                variance of the model for the given input 'x'
            var_mean (torch.Tensor)
                variance of the mean of the individual sub-models (measure for epistemic uncertainty)

        """

        means = []
        stds = []
        for i in range(self.num_models):
            model = getattr(self, 'model_' + str(i))
            mean, std = model(x)
            means.append(mean)
            stds.append(std)
        means = torch.stack(means)
        mean = means.mean(dim=0)
        stds = torch.stack(stds)
        mean_var = stds.pow(2).mean(dim=0) # mean of the variances
        var_mean = means.pow(2).mean(dim=0) - mean.pow(2)  # variance of means
        variance = mean_var + var_mean # total variance of the model
        return mean, variance, var_mean

    def train_ensemble(self, dataset, epochs, loss='nll', lr=1e-3, weight_decay=1e-5):
        """
        Train the ensemble model.

        Args:
            dataset (torch.utils.data.DataSet):
                data set containing the input and target data
            epochs (int):
                number of epochs
            loss (str):
                'nll': training using the negative log-likelihood
                'mse': training using the mean-squared-error
            lr (float):
                learning rate for gradient descent
            weight_decay (float):
                weight-decay for regularization of parameters

        """

        # initialize optimizers
        for i in range(self.num_models):
            model = getattr(self, 'model_' + str(i))
            model.optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(epochs):
            # shuffle data
            for (xb, yb) in dataset.get_batches():
                losses = self._train_ensemble_step(xb, yb, loss)
            if epoch == 0:
                print('inital loss: ', np.mean(losses))
            else:
                print('epoch ',epoch,  ' loss: ', np.mean(losses))
        print('final loss: ', np.mean(np.array(losses)))
        pass



    def _train_ensemble_step(self, x, y, loss_fnc):
        """
        Take one step of gradient descent, when training the deep ensemble.

        Args:
            x (torch.Tensor):
                input tensor
            y (torch.Tensor):
                training target tensor
            loss_fnc (str):
                loss function used for training
                    'nll': training using the negative log-likelihood
                    'mse': training using the mean-squared-error

        Returns:
            losses (numpy.ndarray):
                array of losses of the individual models in the ensemble
        """
        losses = []
        for i in range(self.num_models):
            model = getattr(self, 'model_' + str(i))
            loss = self._train_submodel_step(x, y, model, loss_fnc)
            losses.append(loss)
        losses = np.array(losses)
        return losses


    def _train_submodel_step(self, x, y, model, loss_fnc='nll'):
        """
        Take one step of gradient descent on an individual submodel of the deep ensemble.

        Args:
            x (torch.Tensor):
                input tensor
            y (torch.Tensor):
                training target tensor
            model (torch.nn.module):
                submodel of the deep ensemble that is trained
            loss_fnc (str):
                loss function used for training
                    'nll': training using the negative log-likelihood
                    'mse': training using the mean-squared-error

        Returns:
            loss (float):
                training loss
        """

        # todo: check if requires_grad = True is necessary
        x.requires_grad = True
        model.optimizer.zero_grad()
        mean, var = model(x)
        if loss_fnc == 'nll':
            loss = NLLloss(y, mean, var)
        elif loss_fnc == 'mse':
            loss = nn.MSELoss(y, mean)
        loss.backward()

        # adversarial training
        if self.adversarial:
            x_prime = x + self.eps * torch.sign(x.grad)  # create adversarial examples
            mean, var = model(x)
            mean_prime, var_prime = model(x_prime) # adversarial
            loss = NLLloss(y, mean, var) + NLLloss(y, mean_prime, var_prime) # combined loss
            loss.backward() # perform backprop

        # perform one step of gradient descent
        model.optimizer.step()
        x.requires_grad = False
        return loss.item()

    def export_model(self, file):
        raise NotImplementedError

    def import_model(self, file):
        raise NotImplementedError

class StateSpaceModelDeepEnsemble(DeepEnsemble):
    """
    Class that implements a state-space model deep ensemble (SSM-DE).

    y = f_theta(x, u), where y is either the right-hand side of an ODE or difference equation

    If the model represents a second order system, i.e. a mechanical system, half of the SSM is composed of definitions.
    Therefore only one half of the SSM needs to be modeled by a neural network, the rest of it can be hard-coded.

    For example consider a system with two states x = (x1, x2)^T and input u, where x2 is the velocity of x1. Since we know the
    relation between x1 and x2 a-priori, only the nonlinear term g(x, u) has to represented by a neural network.

        dx1/dt = x2
        dx2/dt = g(x, u)

    with y = (dx1dt, dx2dt)^T = (y1, y2)^T

    To enable this behaviour, set the argument "second_order".
    The output dimension of the neural network is always 0.5*dim(y).

    """
    def __init__(self, state_dim, control_dim, num_models,
                 second_order=False,
                 hidden_layers=[32, 32, 32],
                 activation='relu',
                 std_max=None,
                 sparse=False,
                 adversarial=False,
                 eps=0.001):
        """

        Args:
            state_dim (int):
                state dimension
            control_dim (int):
                control input dimension
            num_models (int):
                number of models in the SSM-DE
            second_order (bool):
                if 'True', the model is split into two parts
            hidden_layers (list):
                layer structure of MLP: [5, 5] (2 hidden layer with 5 neurons)
            activation (str)
                activation function used ('relu', 'tanh', 'sigmoid' or 'swish')
            std_max (torch.Tensor):
                maximum standard deviation with shape (outputs,)
            sparse (bool):
                if 'True', for each output dimension a single MLP is used
            std_max (torch.Tensor):
                maximum values of the standard deviation
            adversarial (bool):
                if 'True', adverserial training is performed
            eps (float):
                parameter for adverserial training
        """
        self.second_order = second_order
        self.state_dim = state_dim
        self.control_dim = control_dim
        if second_order and state_dim % 2 == 0:
            output_dim = int(state_dim/2)
        else:
            output_dim = state_dim

        super(StateSpaceModelDeepEnsemble, self).__init__(num_models=num_models,
                                                          inputs=state_dim+control_dim,
                                                          outputs=output_dim,
                                                          hidden_layers=hidden_layers,
                                                          activation=activation,
                                                          std_max=std_max,
                                                          adversarial=adversarial,
                                                          eps=eps)

    def state_equation(self, state, control):
        """
        State equation that can be used to model an ODE or difference equation.

        Args:
            state (numpy.ndarray, torch.Tensor):
                state vector
            control (numpy.ndarray, torch.Tensor):
                control input vector

        Returns:
            y (numpy.ndarray):
                output of the state equation

        """
        state_t = torch.tensor(state, dtype=torch.float32)
        control_t = torch.tensor(control, dtype=torch.float32)
        mean, var, var_mean = self.forward(torch.cat((state_t, control_t)).reshape(1, self.inputs))
        y = mean.detach().numpy()[0]
        if self.second_order:
            y = np.concatenate([state[self.outputs:], y])
        return y

    def ode(self, t, state, control):
        """
        Ordinary-differential-equation (ODE), that can be used in an IVP-solver (scipy.integrate.solve_ivp).

        Args:
            t (float):
                time, needed by the IVP solver, but the ODE is not time-dependent
            state (numpy.ndarray, torch.Tensor):
                state vector
            control (numpy.ndarray, torch.Tensor):
                control input vector

        Returns:
            y (numpy.ndarray):
                output of the ODE

        """
        y = self.state_equation(state, control)
        return y


