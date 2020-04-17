import torch
import numpy as np
import matplotlib.pyplot as plt

from probecon.neural_networks.deep_ensemble import DeepEnsemble
from probecon.data.dataset import SimpleDataSet

torch.manual_seed(0)
np.random.seed(0)

"""
Reimplementation of the toy example in 


'S. S. Sahoo et al. - Learning Equations for Extrapolation and Control'

Section 4.1 - Learning formulas with divisions

"""

def training_data_set(points=10000):
    xx1 = np.array([[np.random.uniform(-1., 1.)] for _ in range(points)])
    xx2 = np.array([[np.random.uniform(-1., 1.)] for _ in range(points)])
    yy = torch.tensor([np.sin(np.pi*x1)/(x2**2 + 1) + np.random.normal(0, 0.01) for (x1, x2) in zip(xx1, xx2)], dtype=torch.float32)
    xx1 = torch.tensor(xx1, dtype=torch.float32)
    xx2 = torch.tensor(xx2, dtype=torch.float32)
    xx = torch.cat([xx1, xx2], dim=1)
    return xx, yy

xx, yy = training_data_set() # generate data set of 20 samples
data = SimpleDataSet(xx, yy, batch_size=20, shuffle=True)

# plot data
x = np.linspace(-6., 6., 100).reshape(100, 1)
y = np.sin(np.pi*x)/(x**2 + 1)
plt.plot(x, y, 'k-', label=r'ground truth: $y=\frac{\sin(\pi x_1)}{x^2+1}$')
plt.grid()
plt.xlabel(r'$x_1=x_2=x$')
plt.ylabel(r'$y$')
plt.legend()
plt.ylim([-2,2])
plt.show()

# train ensemble model
epochs = 40
ensemble = DeepEnsemble(num_models=5, inputs=2, outputs=1, hidden_layers=[20, 20, 20, 20], activation='swish', adversarial=False)
ensemble.train_ensemble(data, epochs, loss_fnc='nll', lr=1e-3, weight_decay=1e-4)

# plot ensemble model output
plt.plot(x, y, 'k-', label=r'ground truth: $y=\frac{\sin(\pi x_1)}{x^2+1}$')
x_input = torch.tensor(x@np.ones((1, 2)), dtype=torch.float32)
mean, std, std_mean = ensemble(x_input)
mean = mean.detach().numpy()
std = std.detach().numpy()
mean_plot = plt.plot(x, mean)
color = mean_plot[0].get_color()
plt.fill_between(x.reshape(100,), (mean-3*std).reshape(100,), (mean+3*std).reshape(100,),alpha=0.2, color=color)
plt.fill_between(x.reshape(100,), (mean-std).reshape(100,), (mean+std).reshape(100,),alpha=0.3, color=color)
plt.grid()
plt.xlabel(r'$x_1=x_2=x$')
plt.ylabel(r'$y$')
plt.legend()
plt.ylim([-2,2])
plt.show()