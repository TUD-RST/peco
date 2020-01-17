import numpy as np
from probecon.helpers.pygent_helpers import PygentEnvWrapper
from probecon.helpers.mpctools_helpers import MPCToolsEnvWrapper
from probecon.system_models.cart_pole import CartPole

from pygent.algorithms.ilqr import iLQR

class TrajectoryOptimization(object):
    def __init__(self, environment, sim_time, final_cost=None, algorithm='ilqr'):
        self.sim_time = sim_time
        self.algorithm = algorithm
        if algorithm == 'ilqr':
            self.env = PygentEnvWrapper(environment)
            self.ocp = iLQR(self.env, self.sim_time, self.env.dt, path=None, fcost=final_cost, constrained=True)
        elif algorithm == 'collocation':
            self.ocp = MPCToolsEnvWrapper(env, sim_time)

    def solve(self):
        if self.algorithm == 'ilqr':
            self.ocp.run_optim()
            sol = {'x_sim':self.ocp.xx, 'u_sim': self.ocp.uu, 'J_f': self.ocp.cost,
                   'kk': self.ocp.kk,
                   'K': self.ocp.KK,
                   'alpha': self.ocp.current_alpha}
        else:
            sol = self.ocp.solve()
        return sol




if __name__ == '__main__':
    def c_k(x, u):
        x2, x1, x4, x3 = x
        u1, = u
        c = 0.5 * x1 ** 2 + 3 * x2 ** 2 + 0.02 * x3 ** 2 + 0.05 * x4 ** 2 + 0.05 * u1 ** 2
        return c


    def c_N(x):
        x2, x1, x4, x3 = x
        c = 100 * x1 ** 2 + 100 * x2 ** 2 + 10 * x3 ** 2 + 10 * x4 ** 2
        return c


    init_state = np.array([np.pi, 0, 0, 0])
    t = 5
    dt = 0.02

    env = CartPole(time_step=dt, init_state=init_state, cost_function=c_k, part_lin=True)

    traj_opt = TrajectoryOptimization(env, t, final_cost=c_N)

    sol = traj_opt.solve()



