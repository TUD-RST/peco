import numpy as np
from probecon.control.trajectory_optimization import TrajectoryOptimization
from probecon.system_models.cart_pole import CartPole
import matplotlib.pyplot as plt
def ode_error(t, state, control):
    return np.array([np.sin(state[0]), 0., 0., -0.05*control[0]])
    # Todo:
    # realistisches Problem -> Fehlerterm in den Geschwindigkeitskomponenten

def c_k(x, u):
    x2, x1, x4, x3 = x
    u1, = u
    c = 1 * x1 ** 2 + 2 * x2 ** 2 + 0.02 * x3 ** 2 + 0.05 * x4 ** 2 + 0.05 * u1 ** 2
    return c


def c_N(x):
    x2, x1, x4, x3 = x
    c = 100 * x1 ** 2 + 100 * x2 ** 2 + 100 * x3 ** 2 + 100 * x4 ** 2
    return c


sim_time = 3.

env_real = CartPole(cost_function=c_k)
env_model = CartPole(cost_function=c_k, ode_error=ode_error)

traj_opt = TrajectoryOptimization(env_model, sim_time, final_cost=c_N)

sol = traj_opt.solve()

for control in sol['u_sim']:
    env_model.step(control)
    env_model.render()
for control in sol['u_sim']:
    env_real.step(control)
    env_real.render()

env_model.plot()
plt.show()
env_real.plot()
plt.show()

