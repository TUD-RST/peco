import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class StateSpaceEnv(gym.Env):
    """State-space environment.

    state_dimension (int):
    control_dimension (int):
    ode (function): ODE, right hand sight of the differential equation
    init_state (ndarray):



    """
    def __init__(self, state_dim, control_dim, ode, time_step, init_state, goal_state=None, state_cost=None, control_cost=None):

        self.state_dim = state_dim
        self.control_dim = control_dim
        # create spaces
        control_bounds = np.inf * np.ones(control_dim)
        state_bounds = np.inf * np.ones(state_dim)
        self.control_space = spaces.Box(-control_bounds, control_bounds)
        self.state_space = spaces.Box(-state_bounds, state_bounds)

        # test ODEs output using the inital state and zero as input
        assert(ode(0, init_state, np.zeros(control_dim)).shape == (state_dim,))
        # init ODE
        self.ode = ode  # ODE(t, state, control)

        if not isinstance(time_step, float):
            raise TypeError("'time_step' has to be a float")
        self.time_step = time_step

        # set initial state value
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
        

        self.seed()

        # initialize trajectory (time_steps, state/control dim)
        self.trajectory = {'time': np.array([0]), 'states': np.stack([init_state]), 'controls': None}

    def step(self, control):
        if control.shape != (self.control_dim,):
            raise AssertionError("'control' has to be an array with shape '(control_dim,)'")
        self.old_state = self.state
        self.state = self._simulation(self.state)
        self._append_trajectory(self.state, control)

        reward = -self._eval_cost(self.state, control)
        done = False
        return self.state, reward, done, {}

    def reset(self):
        self.old_state = None
        self.state = self.init_state
        return self.state

    def render(self):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _set_state(self, state):
        self.old_state = None
        self.state = state
        return self.state

    def _cost_init(self, state_cost, control_cost):
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

    def _append_trajectory(self, state, control):
        # append state
        states = self.trajectory['states']
        self.trajectory['states'] = np.stack([states, state])

        # append control input
        controls = self.trajectory['controls']
        if controls == None:
            self.trajectory['control inputs'] = np.stack([control])
        else:
            self.trajectory['control inputs'] = np.stack([controls, control])

        # append time
        time = self.trajectory['time']
        self.trajectory['time'] = np.append(time, time[-1]+self.time_step)
        pass
