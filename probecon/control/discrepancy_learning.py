import torch
import numpy as np
import copy

from probecon.nn_models.deep_ensemble import StateSpaceModelDeepEnsemble
from probecon.data.dataset import TransitionDataSet
from probecon.control.trajectory_optimization import TrajectoryOptimization
from probecon.system_models.pendulum import Pendulum

class DiscrepancyLearner(object):
    """
    Class that implements a discrepancy model learning algorithm.

    Assume the following discrete dynamics:

        x[k+1] = x[k] + f(x[k], u[k]) + h(x[k], u[k])

    with state 'x', control 'u', time step 'k', the dynamics model 'f' and an unknown error term 'h'.

    The error term 'h', which describes the discrepancy between the system model 'f' and the real dynamics of the
    environment is modeled by a deep ensemble.

    """
    def __init__(self, real_environment, model_environment, sim_time,
                 alpha=100.,
                 terminal_cost=None,
                 init_opt=False,
                 render=False,
                 second_order=False,
                 sparse=False,
                 std_max=None):
        """

        Args:
            real_environment (probecon.system_models.StateSpaceEnv):
                environment that represents the test bench / real environment
            model_environment (probecon.system_models.StateSpaceEnv):
                environment that represents the model that is used in the controller design
            sim_time (float):
                time horizon of the trajectory optimization
            alpha (float):
                parameter for the discrepancy model
            terminal_cost (function):
                terminal cost at the last time step
            init_opt (bool):
                if 'True', initial trajectory optimization is performed
            render (bool):
                if 'True', environements are rendered
            second_order (bool):
                if 'True', a sparse model (with fewer outputs) is learned
            sparse (bool):
                if 'True', for each output dimension a single MLP is used
            std_max (torch.Tensor):
                maximum values of the standard deviation

        """

        self.render = render
        self.real_environment = real_environment
        self.model_environment = model_environment
        self.sim_time = sim_time
        self.alpha = alpha
        self.terminal_cost = terminal_cost

        state_dim = self.model_environment.state_dim
        control_dim = self.model_environment.control_dim
        if std_max is None:
            std_max = torch.zeros(state_dim)
        self.deep_ensemble = StateSpaceModelDeepEnsemble(num_models=5,
                                                        hidden_layers=[30, 30, 30],
                                                        state_dim=state_dim,
                                                        control_dim=control_dim,
                                                        second_order=second_order,
                                                        sparse=sparse,
                                                        std_max=std_max)

        self.traj_opt = TrajectoryOptimization(self.model_environment, self.sim_time, terminal_cost=self.terminal_cost)
        self.model_environment.set_ode_error(self._discrepancy_model)
        if init_opt:
            print('Initial trajectory optimization')
            self._solve_ocp()
        #self.model_environment.set_ode_error(self._discrepancy_model)
        self.dataset = TransitionDataSet(state_dim, control_dim,
                                         batch_size=512,
                                         shuffle=True,
                                         type='continuous',
                                         state_eq=self.model_environment.ode,
                                         second_order=second_order)
        self.sim_time = sim_time
        self.time_steps = np.arange(0., sim_time+self.model_environment.time_step, self.model_environment.time_step)
        pass

    def _solve_ocp(self):
        """
        Solve the optimal control problem based on the model

        Returns:
            opt_control (numpy.ndarray):
                optimal control trajectory

        """

        self.model_environment.reset()
        self.traj_opt.__init__(self.model_environment, self.sim_time, terminal_cost=self.terminal_cost)
        sol = self.traj_opt.solve()
        # reset the solver
        #self.traj_opt = copy.copy(self.traj_opt_init) # todo: implement a reset method
        self.opt_control = sol['u_sim']

        # rollout the trajectory on the model
        self.model_environment.reset()
        if self.render:
            print('Rendering optimized trajectory on the model!')
            for control in self.opt_control:
                self.model_environment.step(control)
                self.model_environment.render()
            self.model_environment.close()
        return self.opt_control

    def _rollout_episode(self, controls):
        """
        Rollout a control trajectory on the real environment.

        Args:
            controls (numpy.ndarray):
                control trajectory

        Returns:
            trajectory (dict):
                trajectory dictionary with the keys: 'old_states', 'controls', 'states', 'time' and 'time_steps'

        """
        self.real_environment.reset()
        for control in controls:
            self.real_environment.step(control)
            if self.render:
                self.real_environment.render()
        self.real_environment.close()
        return self.real_environment.trajectory

    def _rollout_random_episode(self):
        """
        Rollout a random control trajectory on the real environment-

        Returns:
            trajectory (dict):
                trajectory dictionary with the keys: 'old_states', 'controls', 'states', 'time' and 'time_steps'

        """
        self.real_environment.reset()
        for _ in self.time_steps:
            self.real_environment.random_step()
            if self.render:
                self.real_environment.render()
        self.real_environment.close()
        return self.real_environment.trajectory


    def _train_discrepancy_model(self,
                                epochs=60,
                                loss='nll',
                                lr=1e-3,
                                weight_decay=5e-4):
        """
        Train the discrepancy model.

        Args:
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

        # todo: implement a split between training, test and validation data
        self.deep_ensemble.train_ensemble(self.dataset, epochs=epochs, loss=loss, lr=lr, weight_decay=weight_decay)
        pass

    def _discrepancy_model(self, time, state, control):
        """
        Discrepancy model of the form:
            exp(-alpha*mean_var)*mean,
        where 'mean' is the mean of the deep ensemble,
        'mean_var' is the variance of the individual submodels mean output and captures the epistemic uncertainty and
        alpha, which is a scalar parameter.

        If the epistemic uncertainty, namely 'mean_var' is high, the term exp(-alpha*mean_var) gets smaller and
        the 'weighted_output' approaches '0.0'. If it is low, the 'weighted_output' approaches the value of 'mean'

        Args:
            time (float):
                time is needed by the simulator
            state (numpy.ndarray):
                state vector
            control (numpy.ndarray):
                control input vector

        Returns:
            weighted_output (numpy.ndarray):
                weighted output as described above
        """
        mean, variance, mean_var = self.deep_ensemble.state_eq(state, control)
        weighted_output = (torch.exp(-self.alpha*mean_var)*mean).detach().numpy()[0]
        return weighted_output



    def main(self, epsiodes):
        """
        Main loop of the algorithm

        Args:
            epsiodes (int):
                number of episodes

        """
        for epsiode in range(epsiodes):
            print('Started episode {}!'.format(epsiode))
            if epsiode < 0:
                # 1) apply the control trajectory to the system
                print('1) Random ollout!')
                trajectory = self._rollout_random_episode()

                # 2) add the resulting trajectory to the data set
                print('2) Data aggregation!')
                self.dataset.add_trajectory(trajectory)
            else:
                # 1) compute control trajectory
                print('1) Trajectory optimization!')
                controls = self._solve_ocp() # solve an optimal control problem

                # 2) apply the control trajectory to the system
                print('2) Rollout!')
                trajectory = self._rollout_episode(controls)

                # 3) add the resulting trajectory to the data set
                print('3) Data aggregation!')
                self.dataset.add_trajectory(trajectory)

                # 4) train the discrepancy model
                print('4) Training!')
                #self._train_discrepancy_model()
        pass


if __name__ == '__main__':
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
    sim_time = 5.0
    time_step = 0.02

    env = Pendulum(time_step=time_step, init_state=init_state, cost_function=c_k, ode_error=ode_error)

    mod_env = Pendulum(time_step=time_step, init_state=init_state, cost_function=c_k)

    learner = DiscrepancyLearner(env, mod_env, sim_time=sim_time, terminal_cost=c_N, render=True)

    learner.main(10)