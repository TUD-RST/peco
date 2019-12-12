import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from scipy.integrate import solve_ivp
import pickle

class StateSpaceEnv(gym.Env):
    """State-space environment.

    state_dimension (int):
    control_dimension (int):
    ode (function): ODE, right hand sight of the differential equation
    init_state (ndarray):

    """
    def __init__(self, state_dim, control_dim, ode, time_step, init_state,
                 goal_state=None,
                 state_cost=None,
                 control_cost=None,
                 state_bounds=None,
                 control_bounds=None):

        self.state_dim = state_dim
        self.control_dim = control_dim
        # create spaces
        if isinstance(state_bounds, type(None)):
            state_bounds = np.inf * np.ones(state_dim)
        if isinstance(control_bounds, type(None)):
            control_bounds = np.inf * np.ones(control_dim)

        self.state_space = spaces.Box(-state_bounds, state_bounds)
        self.control_space = spaces.Box(-control_bounds, control_bounds)

        # test ODEs output using the inital state and zero as input
        assert(ode(0, init_state, np.zeros(control_dim)).shape == (state_dim,))
        # init ODE
        self.ode = ode  # ODE(t, state, control)

        if not isinstance(time_step, float):
            raise TypeError("'time_step' has to be a float")
        self.time_step = time_step

        # set initial state value
        # todo: init state can be callable to allow for distributions x0~p(x0)
        if init_state.shape == (state_dim,):
            self.init_state = init_state
        else:
            raise AssertionError("'init_state' has to be an array with shape '(state_dim,)'")

        # goal state
        if isinstance(goal_state, type(None)):
            # default goal state is the zero vector
            self.goal_state = 0*init_state
        else:
            if goal_state.shape == (state_dim, ):
                self.goal_state = goal_state
            else:
                raise AssertionError("'goal_state' has to be an array with shape '(state_dim,)'")

        # init state variables
        self.old_state = None
        self.state = init_state

        # initialize cost
        self._cost_init(state_cost, control_cost)
        
        # seeding
        self.seed()

        # initialize trajectory (time_steps, state/control dim)
        self.trajectory = {'time': np.array([0]), 'states': np.stack([init_state]), 'controls': None}

        self.viewer = None

    def step(self, control):
        """ Do one step in the environment.

               Args:
                   state (ndarray): State vector with shape (state_dim, )
                   control (ndarray): Control input vector with shape (control_dim, )

               Returns:
                   state (ndarray): New state after taking the step
                   reward (float): Reward (-cost) for taking the step
                   done (bool):

               """
        if control.shape != (self.control_dim,):
            raise AssertionError("'control' has to be an array with shape '(control_dim,)'")
        self.old_state = self.state
        self.state = self._simulation(control)
        self._append_transition(self.state, control)
        reward = -self._eval_cost(self.state, control)
        done = self._done()
        info = {}
        return self.state, reward, done, info

    def random_step(self):
        if self.control_space.is_bounded():
            random_control = self.control_space.sample()
        else:
            random_control = np.random.uniform(-1., 1., self.control_dim)
        state, reward, done, info = self.step(random_control)
        return state, reward, done, info

    def reset(self):
        """ Reset the environment to it's initial state and delete the trajectory.

        Returns:
            state (ndarray): current state of the system

        """
        self._set_state(self.init_state)
        return self.state


    def render(self):
        """ gym.Env method """
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def save_trajectory(self, filename, path):
        """ Saves the current trajectory to a pickle file.

        Args:
            filename (string): Saving filename
            path (string): Saving path

        """
        with open(path + filename + '.p', 'wb') as open_file:
            pickle.dump(self.trajectory, open_file)
        pass

    def _set_state(self, state):
        """ Deletes the trajectory and sets the environments state to 'state'.

        Args:
            state (ndarray): State

        """
        self.old_state = None
        self.state = state
        # initialize trajectory (time_steps, state/control dim)
        self.trajectory = {'time': np.array([0]), 'states': np.stack([self.init_state]), 'controls': None}
        pass

    def _simulation(self, control):
        """Simulation of the environment for one time step starting in the current state.

        Args:
            control (ndarray): Control input

        Returns:
            state (ndarray): State after simulation
        """
        sol = solve_ivp(lambda t, state: self.ode(t, state, control), (0, self.time_step), self.state, t_eval=[self.time_step])
        state = sol.y.ravel()
        return state

    def _cost_init(self, state_cost, control_cost):
        """ Initialization of the cost.

                Args:
                    state_cost (ndarray): Cost of the state
                    control_cost (ndarray): Cost of the control

                """
        # terms of a quadratic cost
        if isinstance(state_cost, type(None)):
            self.state_cost = np.diag(np.zeros(self.state_dim))
        else:
            if state_cost.ndim == 1 and state_cost.shape[0] == self.state_dim:
                self.state_cost = np.diag(state_cost)
            elif state_cost.ndim == 2 and state_cost.shape[0] == self.state_dim and state_cost.shape[0] == self.state_dim:
                self.state_cost = state_cost
            else:
                raise AssertionError("'state_cost' has to be an array with shape (state_dim,) or (state_dim, state_dim)")
        if isinstance(control_cost, type(None)):
            self.control_cost = np.diag(np.zeros(self.control_dim))
        else:
            if control_cost.ndim == 1 and control_cost.shape[0] == self.control_dim:
                self.control_cost = np.diag(control_cost)
            elif control_cost.ndim == 2 and control_cost.shape[0] == self.control_dim and control_cost.shape[0] == self.control_dim:
                self.control_cost = control_cost
            else:
                raise AssertionError("'control_cost' has to be an array with shape '(control_dim,)' or '(control_dim, control_dim)'")
        pass

    def _eval_cost(self, state, control):
        """ Quadratic cost function.

        Args:
            state (ndarray): State vector with shape (state_dim, )
            control (ndarray): Control input vector with shape (control_dim, )

        Returns:
            cost (float): Cost

        """
        state_diff = state-self.goal_state
        state_cost = state_diff@self.state_cost@state_diff
        control_cost = control@self.control_cost@control
        cost = 0.5*(state_cost + control_cost)
        return cost

    def _append_transition(self, state, control):
        """ Quadratic cost function.

               Args:
                   state (ndarray): State vector with shape (state_dim, )
                   control (ndarray): Control input vector with shape (control_dim, )

               Returns:
                   cost (float): Cost

               """
        # append state
        states = self.trajectory['states']
        self.trajectory['states'] = np.concatenate((states, state.reshape(1, self.state_dim)))

        # append control input
        controls = self.trajectory['controls']
        if controls == None:
            self.trajectory['control inputs'] = control.reshape(1, self.control_dim)
        else:
            self.trajectory['control inputs'] = np.concatenate((controls, control.reshape(1, self.control_dim)))

        # append time
        time = self.trajectory['time']
        self.trajectory['time'] = np.append(time, time[-1]+self.time_step)
        pass

    def _done(self):
        return not self.state_space.contains(self.state)



class Parameters(object):
    "Container Class for parameters."
    pass