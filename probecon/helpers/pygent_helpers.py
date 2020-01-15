from pygent.environments import StateSpaceModel

class PygentEnvWrapper(StateSpaceModel):
    """ Wrapping StateSpaceEnv environment """

    def __init__(self, environment):
        uDim = environment.control_dim
        ode = environment.ode
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
        return False

    def animation(self):
        return NotImplementedError