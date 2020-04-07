import torch
import numpy as np
import matplotlib.pyplot as plt

from probecon.nn_models.deep_ensemble import DeepEnsemble
from probecon.data.dataset import SimpleDataSet

torch.manual_seed(0)
np.random.seed(0)

"""
Reimplementation of the toy example in https://arxiv.org/abs/1612.01474

'B. Lakshminarayanan et al. - Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles'

Section 3.2 - Regression on toy datasets

Note: The results from the paper can not be reproduced exactly.

"""

def data_set(points=20, xrange=(-4, 4), std=3.):
    xx = torch.tensor([[np.random.uniform(*xrange)] for _ in range(points)])
    yy = torch.tensor([[x**3 + np.random.normal(0, std)] for x in xx])
    return xx, yy

xx, yy = data_set(points=20, xrange=(-4, 4), std=3.) # generate data set of 20 samples
data = SimpleDataSet(xx, yy, batch_size=20, shuffle=True)

# plot data set
x = np.linspace(-6, 6, 100).reshape(100, 1)
y = x**3
plt.plot(x, y, 'b-', label='ground truth: $y=x^3$')
plt.plot(xx.numpy(),yy.numpy(),'or', label='data points')
plt.grid()
plt.xlim(-6., 6.)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# train ensemble model (parameters from the paper)
epochs = 40
ensemble = DeepEnsemble(num_models=5, hidden_layers=[100], activation='relu', adversarial=False)
ensemble.train(data, epochs, loss='nll', lr=0.1, weight_decay=0.)


# plot ensemble model output
plt.plot(x, y, 'b-', label='ground truth: $y=x^3$')
plt.plot(xx.numpy(),yy.numpy(),'or', label='data points')
mean, std, std_mean = ensemble(torch.tensor(x).float())
std = std.detach().numpy()
mean = mean.detach().numpy()
mean_plot = plt.plot(x, mean, label='PE (NLL)')
color = mean_plot[0].get_color()
plt.fill_between(x.reshape(100,), (mean-3*std).reshape(100,), (mean+3*std).reshape(100,),alpha=0.2, color=color)
plt.fill_between(x.reshape(100,), (mean-std).reshape(100,), (mean+std).reshape(100,), alpha=0.3, color=color)
plt.grid()
plt.xlim(-6., 6.)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


# train ensemble model (own parameters)

data = SimpleDataSet(xx, yy, batch_size=20, shuffle=True)

epochs = 1000
ensemble = DeepEnsemble(num_models=5, hidden_layers=[30, 30, 30], activation='swish', adversarial=True)
ensemble.train(data, epochs, loss='nll', lr=15e-3, weight_decay=0.0)


# plot ensemble model output
plt.plot(x, y, 'b-', label='ground truth: $y=x^3$')
plt.plot(xx.numpy(),yy.numpy(),'or', label='data points')
mean, std, std_mean = ensemble(torch.tensor(x).float())
std = std.detach().numpy()
mean = mean.detach().numpy()
mean_plot = plt.plot(x, mean, label='PE (NLL)')
color = mean_plot[0].get_color()
plt.fill_between(x.reshape(100,), (mean-3*std).reshape(100,), (mean+3*std).reshape(100,),alpha=0.2, color=color)
plt.fill_between(x.reshape(100,), (mean-std).reshape(100,), (mean+std).reshape(100,), alpha=0.3, color=color)
plt.grid()
plt.xlim(-6., 6.)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()