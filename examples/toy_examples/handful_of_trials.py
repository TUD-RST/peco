import torch
import numpy as np
import matplotlib.pyplot as plt

from peco.neural_networks.deep_ensemble import DeepEnsemble
from peco.data.dataset import SimpleDataSet

torch.manual_seed(0)
np.random.seed(0)

"""
Reimplementation of the toy example in https://arxiv.org/abs/1805.12114


'K. Chua et al. - Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models'

Section A.2 - Fitting PE model to a toy function

"""

def data_set(points=2000, xranges=[(-2*np.pi, -np.pi), (np.pi, 2*np.pi)]):
    xxs = []
    yys = []
    points = int(points/len(xranges))
    for xrange in xranges:
        xx = np.array([[np.random.uniform(*xrange)] for i in range(points)])
        yy = torch.tensor([[np.sin(x) + np.random.normal(0,0.15*np.sqrt(np.abs(np.sin(1.5*x+np.pi/8))))] for x in xx], dtype=torch.float32)
        xx = torch.tensor(xx, dtype=torch.float32)
        yy = yy.reshape(xx.shape)
        xxs.append(xx)
        yys.append(yy)
    xx = torch.cat((xxs))
    yy = torch.cat((yys))
    return xx, yy

xx, yy = data_set() # generate data set of 20 samples
data = SimpleDataSet(xx, yy, batch_size=100, shuffle=True)

# plot data
x = np.linspace(-5*np.pi, 5*np.pi, 100).reshape(100, 1)
y = np.sin(x)
plt.figure(1)
plt.plot(xx.numpy(),yy.numpy(),'xg', label='data points', markersize=3, alpha=0.3)
plt.plot(x, y, 'k-', label='ground truth: $y=\sin(x)$')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.ylim([-2,2])
plt.plot()

# train ensemble model
epochs = 40
ensemble = DeepEnsemble(num_models=5, hidden_layers=[30, 30, 30], activation='swish', adversarial=True)
ensemble.train_ensemble(data, epochs, loss_fnc='nll', lr=5e-3, weight_decay=1e-5)

# plot submodels output
plt.figure(2)
plt.plot(xx.numpy(),yy.numpy(),'xg', label='data points', markersize=3, alpha=0.3)
plt.plot(x, y, 'k-', label='ground truth: $y=\sin(x)$')
means, stds = ensemble._forward_submodels(torch.tensor(x).float())
for i in range(ensemble.num_models):
    mean = means[i].detach().numpy()
    std = stds[i].detach().numpy()
    mean_plot = plt.plot(x, mean, label='GNN (NLL) '+str(i+1))
    color = mean_plot[0].get_color()
    plt.fill_between(x.reshape(100,), (mean-3*std).reshape(100,), (mean+3*std).reshape(100,), alpha=0.2, color=color)
    plt.fill_between(x.reshape(100,), (mean-std).reshape(100,), (mean+std).reshape(100,), alpha=0.3, color=color)
plt.title('Outputs of the networks in the ensemble')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.ylim([-2,2])

# plot ensemble model output
plt.figure(3)
plt.plot(xx.numpy(),yy.numpy(),'xg', label='data points', markersize=3, alpha=0.3)
plt.plot(x, y, 'k-', label='ground truth: $y=\sin(x)$')
mean, std, std_mean = ensemble(torch.tensor(x).float())
mean = mean.detach().numpy()
std = std.detach().numpy()
mean_plot = plt.plot(x, mean)
color = mean_plot[0].get_color()
plt.fill_between(x.reshape(100,), (mean-3*std).reshape(100,), (mean+3*std).reshape(100,),alpha=0.2, color=color)
plt.fill_between(x.reshape(100,), (mean-std).reshape(100,), (mean+std).reshape(100,),alpha=0.3, color=color)
plt.title('Outputs of the ensemble')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.ylim([-2,2])
plt.show()
