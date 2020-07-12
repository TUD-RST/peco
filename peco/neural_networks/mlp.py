import torch
import torch.nn as nn
import torch.nn.functional as F

from peco.helpers.nn_helpers import Swish

class MLP(nn.Module):
    """
    Class that implements a simple multilayer perceptron (MLP) with different activation functions

    """

    def __init__(self,
                 inputs=1,
                 outputs=1,
                 hidden_layers=[100],
                 activation='relu'):
        """

        Args:
            inputs (int):
                number of inputs of the network
            outputs (int):
                number of outputs of the network
            hidden_layers (list):
                layer structure of the MLP: [5, 5] (2 hidden layer with 5 neurons)
            activation (str):
                activation function used ('relu', 'tanh', 'sigmoid' or 'swish')

        """
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
        elif activation == 'swish':
            self.act = Swish()
        else:
            assert ('Use "relu","tanh", "sigmoid" or "swish" as activation.')
        # create linear layers y = Wx + b

        for i in range(self.nLayers + 1):
            setattr(self, 'layer_' + str(i), nn.Linear(self.net_structure[i], self.net_structure[i + 1]))

    def forward(self, x):
        """
        Forward pass

        Args:
            x (torch.Tensor):
                input vector of shape (batch size, inputs)

        Returns:
            y (torch.Tensor):
                output vector of shape (batch size, outputs)

        """
        # connect layers
        for i in range(self.nLayers):
            layer = getattr(self, 'layer_' + str(i))
            x = self.act(layer(x))
        layer = getattr(self, 'layer_' + str(self.nLayers))
        y = layer(x)
        return y

class SparseMLP(nn.Module):
    """
    Class that implements a sparse MLP, which means that every single output dimension
    is treated seperately by an individual MLP

    """
    def __init__(self, inputs=1, outputs=1, hidden_layers=[100], activation='relu'):
        """

        Args:
            inputs (int):
                number of inputs of the network
            outputs (int):
                number of outputs of the network
            hidden_layers (list):
                layer structure of the MLP: [5, 5] (2 hidden layer with 5 neurons)
            activation (str):
                activation function used ('relu', 'tanh', 'sigmoid' or 'swish')

        """
        super(SparseMLP, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.hidden_layers = hidden_layers
        self.net_structure = [inputs, *hidden_layers, outputs]
        for i in range(outputs):
            mlp = MLP(inputs, outputs=1, hidden_layers=hidden_layers, activation=activation)
            setattr(self, 'mlp_' + str(i), mlp)

    def forward(self, x):
        """
        Forward pass

        Args:
            x (torch.Tensor):
                input vector of shape (batch size, inputs)

        Returns:
            y (torch.Tensor):
                output vector of shape (batch size, outputs)

        """
        ys = [getattr(self, 'mlp_' + str(i))(x) for i in range(self.outputs)]
        y = torch.cat(ys, dim=1) # concatenate the individual network outputs
        return y

class GaussianMLP(nn.Module):
    """
    Class that implements a Gaussian MLP which outputs are mean and variance

    """

    def __init__(self,
                 inputs=1,
                 outputs=1,
                 hidden_layers=[100],
                 activation='relu',
                 std_max=None,
                 sparse=True):
        """

        Args:
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
                if 'True', for each input dimension a single MLP is used
        """
        super(GaussianMLP, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.std_max = std_max
        self.sparse = sparse
        if self.sparse:
            self.mlp = SparseMLP(inputs=inputs,
                                 outputs=2*outputs,
                                 hidden_layers=hidden_layers,
                                 activation=activation)
        else:
            self.mlp = MLP(inputs=inputs,
                           outputs=2*outputs,
                           hidden_layers=hidden_layers,
                           activation=activation)

    def forward(self, x):
        """
        Forward pass of the network

        Args:
            x (torch.Tensor):
                input tensor of shape (batch size, inputs)

        Returns:
            mean (torch.Tensor):
                mean tensor of shape (batch size, outputs)
            std (torch.Tensor):
                standard deviation tensor of shape (batch-size, outputs)

        """

        y = self.mlp(x)
        mean, std = torch.split(y, self.outputs, dim=1)
        if self.std_max is not None:
            std = (self.std_max*torch.sigmoid(std))
        else:
            std = F.softplus(std) + 1e-6 # ensure the 'std' is positive
        return mean, std


if __name__ == '__main__':
    net = MLP(inputs=2, outputs=2)
    net2 = SparseMLP(inputs=2, outputs=2)
    net3 = GaussianMLP(inputs=2, outputs=2, sparse=False)
    x = torch.ones((10, 2))
    print(net(x))
    print(net2(x))
    print(net3(x))
