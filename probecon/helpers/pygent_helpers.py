from pygent.environments import StateSpaceModel

class EnvWrapper(StateSpaceModel):
    """ Wrapping StateSpaceEnv environment """

    def __init__(self, environment):
        self.environment = environment
        self.uDim = environment.control_dim
        self.xDim = environment.state_dim
        self.uMax = environment.control_space.high
        super(EnvWrapper, self).__init__(self, environment.ode, self.eval_cost, environment.init_state,
                                         environment.control_dim, environment.time_step)


    def eval_cost(self, state, control, new_state, t, mod):
        c = self.environment._eval_cost(state, control)
        return c


    def terminate(self, x):
        return False

    def animation(self):
        return NotImplementedError

