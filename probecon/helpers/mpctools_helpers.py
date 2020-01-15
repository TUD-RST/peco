from probecon.system_models.core import SymbtoolsEnv
from probecon.system_models.cart_pole import CartPole
import casadi as ca
import numpy as np
import rstmpctools as rstmpc
from rstmpctools.mpc import MPCSolver
from rstmpctools.ocp import OptimalControlProblem

class MPCToolsEnvWrapper(object):
    def __init__(self, environment, sim_time, mpc_horizon=None):
        self.environment = environment
        self.sim_time = sim_time
        self.mpc_horizon = mpc_horizon
        state_dim = self.environment.state_dim
        control_dim = self.environment.control_dim
        params = self.environment.p.__dict__

        x = ca.MX.sym('x', state_dim)
        u = ca.MX.sym('u', control_dim)
        p = ca.MX.sym('p', params.__len__())

        sym_dict = {'x': x, 'u': u, 'p': p}

        p_val_dict = {'p': np.array(list(params.values()))}

        sxup_dict = {}

        repl_dict = {} # replacement dict

        for i, key in enumerate(params.keys()):
            sxup_dict['p_'+str(i)] = key # add parameters
            repl_dict['p['+str(i)+']'] = key
        for i, state in enumerate(self.environment.mod.xx._mat):
            sxup_dict['x_' + str(i)] = state.name  # add parameters
            repl_dict['x[' + str(i) + ']'] = state.name
        for i, control in enumerate(self.environment.mod.uu._mat):
            sxup_dict['u_' + str(i)] = control.name
            repl_dict['u[' + str(i) + ']'] = control.name

        state_eq = self.environment.state_eq_expr
        state_eq_list = list(state_eq)
        state_eq_str_list = [str(item) for item in state_eq_list]

        # replace sympy symbols with casadi symbols
        for key, val in repl_dict.items():
            # for every element of dx/dt replace val with key
            for i, sub_eq in enumerate(state_eq_str_list):
                state_eq_str_list[i] = sub_eq.replace(val, key)

        # replace function calls to CasADi
        casadi_state_eq_str_list = [SympyToCasadi(item) for item in state_eq_str_list]

        # create casadi expression of the rhs
        casadi_expr = []
        for item in casadi_state_eq_str_list:
            casadi_expr.append(eval(item))
        # casadi_expr = [eval(item) for item in casadi_state_eq_str_list]
        x_dot_p = ca.vertcat(*casadi_expr)
        # create an rst-mpctools system
        self.mpctools_env = rstmpc.System(x_dot_p, sym_dict, p_val_dict, sxup_dict)

        self._create_ocp()
        self._create_solver()

    def _create_ocp(self):
        mode = 'custom'

        sys = self.mpctools_env
        x_0 = self.environment.init_state
        uprev = np.zeros(sys.n_u)

        # Set constant state and input reference.
        x_ref = self.environment.goal_state.reshape(1, sys.n_x)
        u_ref = np.zeros((1, sys.n_u))

        # Specify time grid for simulation and discretization via time step size T
        # and final time in seconds t_f.
        dt = self.environment.time_step
        tf = self.sim_time
        tgrid = np.linspace(0, tf, int(tf / dt) + 1)
        assert tgrid[-1] == tf and tgrid[1] - tgrid[0] == dt

        # set lower bounds for state 'x', input 'u' and change of input 'Du'.
        lb = {'x': self.environment.state_space.low,
              'u': self.environment.control_space.low,
              'Du': -np.inf*np.ones(self.environment.control_dim)}

        # set upper bounds for state 'x', input 'u' and change of input 'Du'.
        ub = {'x': self.environment.state_space.high,
              'u': self.environment.control_space.high,
              'Du': np.inf*np.ones(self.environment.control_dim)}

        # cost Matrices
        Q = self.environment.state_cost
        R = self.environment.control_cost

        # create OptimalControlProblem object.
        self.ocp = OptimalControlProblem(sys, mode, x_0, tgrid, x_ref, u_ref,
                                         kwargs={'lb': lb, 'ub': ub, 'uprev': uprev, 'Q': Q, 'R': R})
        return self.ocp

    def _create_solver(self):
        # Specify keyword arguments of MPCSolver constructor.
        if self.mpc_horizon is None:
            receding = 'global'
            N = 2 # needs to be greater 1, but is overwritten because of 'global'
        else:
            receding = 'unltd'
            N = self.mpc_horizon

        kwargs = {'N': N,
                  'receding': receding,
                  'u_guess': None,
                  'x_guess': None,
                  'verbosity': 0}

        # Create MPCSolver object. See constructor docstring for details.
        self.solver = MPCSolver(self.ocp, **kwargs)
        return self.solver

    def solve(self):
        """ Solves the optimal control problem. """
        sol = self.solver.solve()
        return sol

    def plot(self):
        """ Plot the optimal state and control trajectories. """
        self.solver._plot()
        pass


def SympyToCasadi(string):
    """ Converts a string of a Sympy expression (no module prefix) to CasADi. """

    string = string.replace('cos', 'ca.cos')
    string = string.replace('sin', 'ca.sin')
    return string

if __name__ == '__main__':
    env = CartPole()
    sim_time = 3.
    mpc_ocp = MPCToolsEnvWrapper(env, sim_time)
    sol = mpc_ocp.solve()
    mpc_ocp.plot()
    print(sol['sol_status'])
