import torch
import numpy as np

from probecon.nn_models.deep_ensemble import StateSpaceModelDeepEnsemble
from probecon.data.dataset import TransitionDataSet
from probecon.control.trajectory_optimization import TrajectoryOptimization
from probecon.system_models.pendulum import Pendulum

class DiscrepancyLearner(object):
    def __init__(self, real_environment, model_environment, sim_time,
                 final_cost=None,
                 init_opt=True,
                 render=False,
                 second_order=False):
        self.render = render
        self.real_environment = real_environment
        self.model_environment = model_environment
        state_dim = self.model_environment.state_dim
        control_dim = self.model_environment.control_dim
        self.deep_ensemble = StateSpaceModelDeepEnsemble(num_models=5,
                                                    hidden_layers=[32, 32, 32],
                                                    state_dim=state_dim,
                                                    control_dim=control_dim,
                                                    second_order=second_order,
                                                    alpha=100,
                                                    std_max=torch.zeros(state_dim))
        self.traj_opt = TrajectoryOptimization(self.model_environment, sim_time, final_cost=final_cost)
        if init_opt:
            sol = self.traj_opt.solve()
            self.opt_control = sol['u_sim']
        self.model_environment.set_ode_error(self.deep_ensemble.ode)
        self.dataset = TransitionDataSet(state_dim, control_dim,
                                         batch_size=100,
                                         shuffle=True,
                                         type='continuous',
                                         ode=self.model_environment.ode,
                                         second_order=second_order)
        self.sim_time = sim_time
        self.time_steps = np.arange(0., sim_time+self.model_environment.time_step, self.model_environment.time_step)
        pass

    def solve_ocp(self):
        self.model_environment.reset()
        # rollout the optimal trajectory computed for the model
        #for control in self.opt_control:
        #    self.model_environment.step(control)
            #if self.render:
            #    self.model_environment.render()
        #self.model_environment.close()
        # compute the optimal trajectory with the mixed model
        sol = self.traj_opt.solve()
        opt_control = sol['u_sim']
        # rollout the trajectory on the model
        self.model_environment.reset()
        for control in opt_control:
            self.model_environment.step(control)
            #if self.render:
            #    self.model_environment.render()
        #self.model_environment.close()
        return opt_control

    def rollout_episode(self, controls):
        # solve trajectory optimization problem:
        for control in controls:
            self.real_environment.step(control)
            if self.render:
                self.real_environment.render()
        self.real_environment.close()
        return self.real_environment.trajectory

    def rollout_random_episode(self):
        # solve trajectory optimization problem:
        self.real_environment.reset()
        for _ in self.time_steps:
            self.real_environment.random_step()
            if self.render:
                self.real_environment.render()
        self.real_environment.close()
        return self.real_environment.trajectory


    def train_ensemble(self):
        self.deep_ensemble.train_ensemble(self.dataset, 200, loss='nll', lr=1e-3, weight_decay=1e-5)
        #self.deep_ensemble.train_ensemble(self.dataset, 100, loss='nll', lr=1e-4, weight_decay=1e-4)
        #self.deep_ensemble.train_ensemble(self.dataset, 100, loss='nll', lr=1e-5, weight_decay=1e-4)
        pass


    def main(self, epsiodes):
        for epsiode in range(epsiodes):
            print('Started episode {epsiode}!', epsiode)
            if epsiode==0:
                controls = self.opt_control
            else:
                controls = self.solve_ocp()
            #trajectory = self.rollout_random_episode()
            trajectory = self.rollout_episode(controls)
            self.dataset.add_trajectory(trajectory)
            print('Started training !')
            self.train_ensemble()
        pass


if __name__ == '__main__':
    """
    def c_k(x, u):
        x2, x1, x4, x3 = x
        u1, = u
        c = 1.5 * x1 ** 2 + 3 * x2 ** 2 + 0.02 * x3 ** 2 + 0.05 * x4 ** 2 + 0.05 * u1 ** 2
        return c


    def c_N(x):
        x2, x1, x4, x3 = x
        c = 100 * x1 ** 2 + 100 * x2 ** 2 + 10 * x3 ** 2 + 10 * x4 ** 2
        return c

    def ode_error(t, state, control):
        return np.array([np.sin(state[0]), 0., 0., -0.05 * control[0]])


    init_state = np.array([np.pi, 0, 0, 0])
    sim_time = 3.0
    time_step = 0.02

    env = CartPole(time_step=time_step, init_state=init_state, cost_function=c_k, part_lin=True, ode_error=ode_error)

    mod_env = CartPole(time_step=time_step, init_state=init_state, cost_function=c_k, part_lin=True)

    learner = DiscrepancyLearner(env, mod_env, sim_time=sim_time, final_cost=c_N, render=True)

    learner.main(5)
    """
    def c_k(x, u):
        x1, x2 = x
        u1, = u
        c = 1.5 * x1 ** 2 + 0.02 * x2 ** 2 + 0.05 * u1 ** 2
        return c


    def c_N(x):
        x1, x2 = x
        c = 100 * x1 ** 2 + 100 * x2 ** 2
        return c

    def ode_error(t, state, control):
        return np.array([np.sin(state[0]), -0.05 * control[0]])


    init_state = np.array([np.pi, 0])
    sim_time = 3.0
    time_step = 0.02

    env = Pendulum(time_step=time_step, init_state=init_state, cost_function=c_k, ode_error=ode_error)

    mod_env = Pendulum(time_step=time_step, init_state=init_state, cost_function=c_k)

    learner = DiscrepancyLearner(env, mod_env, sim_time=sim_time, final_cost=c_N, render=True)

    learner.main(5)