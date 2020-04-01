from pygent.environments import StateSpaceModel

class PygentEnvWrapper(StateSpaceModel):
    """
    Class of a 'pygent' wrapper for  the StateSpaceEnv class

    This enables to use the algorithms (DDPG, NFQ, iLQR and MBRL) of

    https://github.com/tud-rst/pygent

    """

    def __init__(self, environment):
        """

        Args:
            environment (probecon.StateSpaceEnv):
                environment object that should be wrapped
        """

        uDim = environment.control_dim
        ode = environment.rhs
        cost = self.eval_cost
        x0 = environment.init_state
        dt = environment.time_step
        super(PygentEnvWrapper, self).__init__(ode, cost, x0, uDim, dt)
        self.uMax = environment.control_space.high
        self.environment = environment

    def eval_cost(self, state, control, new_state, t, mod):
        c = self.environment._eval_cost(state, control)
        return c


    def terminate(self, x):
        """
        Determines if the state 'x' is terminal

        Args:
            x (numpy.ndarrray):

        Returns:
            True if 'x' is terminal

        """
        return False

    def animation(self):
        return NotImplementedError