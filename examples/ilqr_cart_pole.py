import numpy as np
from probecon.helpers.pygent_helpers import PygentEnvWrapper
from probecon.system_models.cart_pole import CartPole

from pygent.algorithms.ilqr import iLQR
def c_k(x, u):
    x2, x1, x4, x3 = x
    u1, = u
    c = 0.5 * x1 ** 2 + 3 * x2 ** 2 + 0.02 * x3 ** 2 + 0.05 * x4 ** 2 + 0.05 * u1 ** 2
    return c

def c_N(x):
    x2, x1, x4, x3 = x
    c = 100*x1**2 + 100*x2**2 + 10*x3**2 + 10*x4**2
    return c

init_state = np.array([np.pi, 0, 0, 0])
t = 5
dt = 0.02

env = CartPole(time_step=dt, init_state=init_state, cost_function=c_k, part_lin=True)
env = PygentEnvWrapper(env)
path = '../../results/ilqr/cart_pole/' # path, where results are saved

algorithm = iLQR(env, t, dt, path=path, constrained=True) # instance of the iLQR algorithm
algorithm.run_optim() # run trajectory optimization
