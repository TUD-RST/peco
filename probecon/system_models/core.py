import gym
import os
from gym import spaces
from gym.utils import seeding
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import pickle
import matplotlib.pyplot as plt


class StateSpaceEnv(gym.Env):
    """
    Class for a state-space model environment, that is compatible with OpenAI's Gym.

    """
    def __init__(self, state_dim, control_dim, ode, time_step, init_state,
                 goal_state=None,
                 state_cost=None,
                 control_cost=None,
                 cost_function=None,
                 state_bounds=None,
                 control_bounds=None,
                 ode_error=None):
        """
        Args:
            state_dim (int):
                dimension of the state space
            control_dim(int):
                dimension of the control input space
            ode (function):
                ODE, right hand side of the differential equation
            time_step (float):
                sampling time
            init_state (numpy.ndarray):
                initial state
            goal_state (numpy.ndarray):
                goal state of the environment
            state_cost (numpy.ndarray):
                cost of the state vector
            control_cost (numpy.ndarray):
                cost of the control vector
            cost_function (function):
                explicit cost function (for example a non-quadratic or exponential cost)
            state_bounds (numpy.ndarray):
                box constraints of the state space
            control_bounds (numpy.ndarray):
                box constraints of the control input space
            ode_error (function):
                ODE error, additional termi in right hand side

        """
        self.state_dim = state_dim
        self.control_dim = control_dim
        # create spaces
        if isinstance(state_bounds, type(None)):
            state_bounds = np.inf * np.ones(state_dim)
        if isinstance(control_bounds, type(None)):
            control_bounds = np.inf * np.ones(control_dim)

        self.state_space = spaces.Box(-state_bounds, state_bounds)
        self.control_space = spaces.Box(-control_bounds, control_bounds)

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
            self.goal_state = np.zeros(state_dim)
        else:
            if goal_state.shape == (state_dim, ):
                self.goal_state = goal_state
            else:
                raise AssertionError("'goal_state' has to be an array with shape '(state_dim,)'")

        # test ODEs output using the inital state and zero as input
        assert (ode(0, init_state, np.zeros(control_dim)).shape == (state_dim,))
        self.ode = ode

        # ODE error term
        if ode_error is not None:
            self.set_ode_error(ode_error)
        else:
            self.rhs = ode

        # init state variables
        self.old_state = None
        self.state = init_state

        # initialize cost
        self._cost_init(state_cost, control_cost)

        # seeding
        self.seed()

        # initialize trajectory (time_steps, state/control dim)
        self.trajectory = {'time': np.array([[0]]), 'states': np.stack([init_state]), 'controls': None}

        self.viewer = None

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(1 / time_step)
        }

        # if explicit cost function is given overwrite quadratic cost
        self.cost_function = cost_function
        if callable(cost_function):
            self._eval_cost = cost_function

    def step(self, control):
        """
        Take one step in the environment (forward simulation).

        Args:
            control (numpy.ndarray):
                control input vector with shape (control_dim, )

        Returns:
            state (numpy.ndarray):
                new state after taking the step
            reward (float):
                reward (-cost) for taking the step
            done (bool):
                True, if terminal state is reached
            info (dict):
                empty dict for additional info

        """
        if control.shape != (self.control_dim, ):
            raise AssertionError("'control' has to be an array with shape '(control_dim,)'")
        control = np.clip(control, self.control_space.low, self.control_space.high)
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
        """
        Reset the environment to it's initial state and delete the trajectory.

        Returns:
            state (numpy.ndarray): current state of the system

        """
        self._set_state(self.init_state)
        return self.state


    def render(self):
        """
        gym.Env method for rendering.

        """
        raise NotImplementedError

    def close(self):
        """
        Closes the Gym render viewer.

        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        pass

    def seed(self, seed=None):
        """
        Seed the environment
        Args:
            seed (int):
                seed value

        Returns:
            [seed] (list):
                seed value

        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def plot(self):
        """
        Creates a plot of the state and control trajectories.

        Returns:
            fig (matplotlib.figure.Figure)

        """
        time = self.trajectory['time']
        states = self.trajectory['states']
        controls = self.trajectory['controls']
        fig, axes = plt.subplots(2)
        axes[0].plot(time, states)
        axes[0].legend([r'$x_'+str(i+1)+'$' for i in range(self.state_dim)], loc='upper right')
        axes[0].grid(True)
        axes[1].plot(time[:-1], controls)
        axes[1].legend([r'$u_' + str(i+1) + '$' for i in range(self.control_dim)], loc='upper right')
        axes[1].set_xlabel(r't [s]')
        axes[1].grid(True)
        return fig

    def save_trajectory(self, filename, path):
        """ Saves the current trajectory to a pickle file

        Args:
            filename (str):
                name of the trajectory to be saved
            path (str):
                path of the trajectory to be saved

        """
        with open(path + filename + '.p', 'w') as open_file:
            pickle.dump(self.trajectory, open_file)
        pass

    def _set_state(self, state):
        """
        Deletes the trajectory and sets the environments state to 'state'

        Args:
            state (numpy.ndarray):
                state the environment is reset to

        """
        self.old_state = None
        self.state = state
        # initialize trajectory (time_steps, state/control dim)
        self.trajectory = {'time': np.array([[0]]), 'states': np.stack([self.init_state]), 'controls': None}
        pass

    def _simulation(self, control):
        """
        Simulation of the environment for one time-step starting in the current state

        Args:
            control (numpy.ndarray): Control input

        Returns:
            state (numpy.ndarray): State after simulation
        """
        sol = solve_ivp(lambda t, state: self.rhs(t, state, control), (0, self.time_step), self.state, t_eval=[self.time_step])
        state = sol.y.ravel()
        return state

    def _cost_init(self, state_cost, control_cost):
        """
        Initialization of the cost function

        Args:
            state_cost (numpy.ndarray): Cost of the state
            control_cost (numpy.ndarray): Cost of the control

        """
        if state_cost is None:
            self.state_cost = np.diag(np.ones(self.state_dim))
        else:
            if state_cost.ndim == 1 and state_cost.shape[0] == self.state_dim:
                self.state_cost = np.diag(state_cost)
            elif state_cost.ndim == 2 and state_cost.shape[0] == self.state_dim and state_cost.shape[0] == self.state_dim:
                self.state_cost = state_cost
            else:
                raise AssertionError("'state_cost' has to be an array with shape (state_dim,) or (state_dim, state_dim)")
        if control_cost is None:
            self.control_cost = np.diag(np.ones(self.control_dim))
        else:
            if control_cost.ndim == 1 and control_cost.shape[0] == self.control_dim:
                self.control_cost = np.diag(control_cost)
            elif control_cost.ndim == 2 and control_cost.shape[0] == self.control_dim and control_cost.shape[0] == self.control_dim:
                self.control_cost = control_cost
            else:
                raise AssertionError("'control_cost' has to be an array with shape '(control_dim,)' or '(control_dim, control_dim)'")
        pass

    def _eval_cost(self, state, control):
        """
        Quadratic cost function

        Args:
            state (numpy.ndarray): State vector with shape (state_dim, )
            control (numpy.ndarray): Control input vector with shape (control_dim, )

        Returns:
            cost (float):
                stage cost of the transition

        """
        state_diff = state-self.goal_state
        state_cost = state_diff@self.state_cost@state_diff
        control_cost = control@self.control_cost@control
        cost = 0.5*(state_cost + control_cost)*self.time_step
        return cost

    def _append_transition(self, state, control):
        """
        Append a transtion to the trajectory

        Args:
            state (numpy.ndarray):
                state vector with shape (state_dim, )
            control (numpy.ndarray):
                control input vector with shape (control_dim, )

        """
        # append state
        states = self.trajectory['states']
        self.trajectory['states'] = np.concatenate((states, state.reshape(1, self.state_dim)))

        # append control input
        controls = self.trajectory['controls']
        if controls is None:
            self.trajectory['controls'] = control.reshape(1, self.control_dim)
        else:
            self.trajectory['controls'] = np.concatenate((controls, control.reshape(1, self.control_dim)))

        # append time
        time = self.trajectory['time']
        self.trajectory['time'] = np.append(time, np.array([time[-1]+self.time_step]), axis=0)
        pass

    def _done(self):
        """
        Determines if the current state is out of bounds

        Returns:
            done (bool):
                True, if state is out of bounds

        """
        done = not self.state_space.contains(self.state)
        return done

    def set_ode_error(self, ode_error):
        """
        Setting an error term in the environments ODE to model unknown dynamics
        Args:
            ode_error (function):
                function that describes the error term of the environments ODE

        """
        assert(ode_error(0, self.init_state, np.zeros(self.control_dim)).shape == (self.state_dim,))
        self.ode_error = ode_error
        self.rhs = lambda t, state, control: self.ode(t, state, control) + self.ode_error(t, state, control)
        pass

class SymbtoolsEnv(StateSpaceEnv):
    """
    Class for mechanical systems (environments), that were derived with the symbtools module.

    """
    def __init__(self, mod_file, params, time_step,
                 init_state=None,
                 goal_state=None,
                 state_cost=None,
                 control_cost=None,
                 cost_function=None,
                 state_bounds=None,
                 control_bounds=None,
                 part_lin=False,
                 ode_error=None):
        """
        Args:
            mod_file (str):
                filename of the pickle file, where the model container was dumped into
            params (dict):
                system parameters
            time_step (float):
                duration of one time-step
            init_state:
                initial state of the environment
            goal_state (numpy.ndarray):
                goal state of the environment
            state_cost (numpy.ndarray):
                cost of the state vector
            control_cost (numpy.ndarray):
                cost of the control vector
            cost_function (function):
                explicit cost function (for example a non-quadratic or exponential cost)
            state_bounds (numpy.ndarray):
                box constraints of the state space
            control_bounds (numpy.ndarray):
                box constraints of the control input space
            part_lin (bool):
                True, if the partial-linearized form of the dynamics should be used
            ode_error (function):
                ODE error, additional termi in right hand side

        """

        # parameters:
        self.p = params

        # load mod file
        package_file_directory = os.path.dirname(os.path.abspath(__file__))
        if mod_file.find('/') == -1:
            # if file only contains the filename, use the packages model fiels
            mod_file = os.path.join(package_file_directory, 'symbtools_models', mod_file)
        with open(mod_file, 'rb') as open_file:
            self.mod =  pickle.load(open_file)
        if part_lin:
            if self.mod.ff is None:
                raise NotImplementedError
            self.state_eq_expr = self.mod.ff + self.mod.gg*self.mod.uu
        else:
            self.state_eq_expr = self.mod.f + self.mod.g*self.mod.uu #self.mod.state_eq
        try:
            import sympy_to_c as sp2c
            self.state_eq_fnc = sp2c.convert_to_c((*self.mod.xx, *self.mod.uu, *self.mod.params), self.state_eq_expr, use_exisiting_so=False)
            print('c code')
        except:
            self.state_eq_fnc = sp.lambdify((*self.mod.xx, *self.mod.uu, *self.mod.params), self.state_eq_expr, modules="numpy")  # creating a callable python function

        state_dim = self.mod.xx.__len__()
        control_dim = self.mod.uu.__len__()

        # check if parameters are aligned
        for key, param in zip(self.p.__dict__.keys(), self.mod.params):
            assert(key==str(param))

        ode = lambda t, x, u: self.state_eq_fnc(*x, *u, *self._params_vals()).ravel()

        # compute jacobians (A and B matrix)
        ode_state_jac = sp.lambdify((*self.mod.xx, *self.mod.uu, *self.mod.params),
                                    self.state_eq_expr.jacobian(self.mod.xx), modules="numpy")
        self.ode_state_jac = lambda x, u: ode_state_jac(*x, *u, *self._params_vals())
        ode_control_jac = sp.lambdify((*self.mod.xx, *self.mod.uu, *self.mod.params),
                                    self.state_eq_expr.jacobian(self.mod.uu), modules="numpy")
        self.ode_control_jac = lambda x, u: ode_control_jac(*x, *u, *self._params_vals())

        super(SymbtoolsEnv, self).__init__(state_dim, control_dim, ode, time_step, init_state,
                                           goal_state=goal_state,
                                           state_cost=state_cost,
                                           control_cost=control_cost,
                                           cost_function=cost_function,
                                           state_bounds=state_bounds,
                                           control_bounds=control_bounds,
                                           ode_error=ode_error)

    def _params_vals(self):
        """
        Determine the parameter values of the system for evaluation in the system dynamics

        Returns:
            params_vals (list):
                list of parameter values

        """
        params_vals = list(self.p.__dict__.values())
        return params_vals

class Parameters(object):
    """
    Container class for parameters

    """