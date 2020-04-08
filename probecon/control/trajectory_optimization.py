import numpy as np
from probecon.helpers.pygent_helpers import PygentEnvWrapper
from probecon.helpers.mpctools_helpers import MPCToolsWrapper
from probecon.system_models.cart_pole import CartPole

from gym.wrappers.monitoring.video_recorder import VideoRecorder

from pygent.algorithms.ilqr import iLQR

class TrajectoryOptimization(object):
    """
    Class that wraps different trajectory optimization solvers.

    """
    def __init__(self, environment, sim_time,
                 terminal_cost=None,
                 algorithm='ilqr'):
        """

        Args:
            environment (probecon.system_models.core.StateSpaceEnv):
                environment to which the trajectory optimization is applied to
            sim_time (float):
                length of the trajectory
            terminal_cost (function):
                terminal cost function of the last time step
            algorithm (str):
                'ilqr':
                    use the iLQR algorithm from https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf
                'collocation':
                    use a collocation based solver from https://bitbucket.org/rawlings-group/mpc-tools-casadi/
        """
        self.sim_time = sim_time
        self.algorithm = algorithm
        if algorithm == 'ilqr':
            self.env = PygentEnvWrapper(environment)
            self.ocp = iLQR(self.env, self.sim_time, self.env.dt,
                            fcost=terminal_cost,
                            constrained=True,
                            maxIters=200)
        elif algorithm == 'collocation':
            self.env = environment
            self.ocp = MPCToolsWrapper(self.env, sim_time)

    def solve(self):
        """
        Solve the trajectory optimization problem.

        Returns:
            sol (dict):
                solution dictionary containing:
                    'x_sim': optimal state trajectory
                    'u_sim': optimal control trajectory
                    'J_f': final cost of the trajectory
                    todo: these are not all entries

        """
        if self.algorithm == 'ilqr':
            self.ocp.reset()
            self.ocp.environment.reset(self.ocp.environment.x0)
            self.ocp.run_optim()
            sol = {'x_sim':self.ocp.xx, 'u_sim': self.ocp.uu, 'J_f': self.ocp.cost,
                   'k': self.ocp.kk,
                   'K': self.ocp.KK,
                   'alpha': self.ocp.current_alpha}
        else:
            sol = self.ocp.solve()
        return sol




if __name__ == '__main__':
    def c_k(x, u, mod):
        x2, x1, x4, x3 = x
        u1, = u
        c = 2*x1**2 + 0.01*x2** 2 + 0.01*x3**2 + 0.01*x4**2 + 0.01*u1**2
        return c


    def c_N(x):
        x2, x1, x4, x3 = x
        c = 100 * x1 ** 2 + 100 * x2 ** 2 + 10 * x3 ** 2 + 10 * x4 ** 2
        return c


    init_state = np.array([np.pi, np.pi, 0, 0])
    t = 6.
    dt = 0.02

    env = CartPole(time_step=dt, init_state=init_state, cost_function=c_k, part_lin=True)

    traj_opt = TrajectoryOptimization(env, t, terminal_cost=c_N)

    sol = traj_opt.solve()

    vid = VideoRecorder(env, 'recording/video.mp4')

    for u in sol['u_sim']:
        env.step(u)
        #env.render()
        vid.capture_frame()
    vid.close()
    #env.close()



