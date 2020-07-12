import unittest
import numpy as np
from peco.system_models.core import StateSpaceEnv
from peco.system_models.cart_triple_pole import CartTriplePole
class TestStateSpaceEnv(unittest.TestCase):

    def test_init_correct_input(self):
        ode = lambda t, state, control: np.array([-state[0] + control[0], -state[1]])
        state_dim = 2
        control_dim = 1
        time_step = 0.01
        init_state = np.array([1, -1])
        env = StateSpaceEnv(state_dim, control_dim, ode, time_step, init_state)
        state_cost = np.array([1, 1])
        control_cost = np.array([1])
        env = StateSpaceEnv(state_dim, control_dim, ode, time_step, init_state, goal_state=init_state, state_cost=state_cost, control_cost=control_cost)
        return env

    def test_init_wrong_input(self):
        ode = lambda t, state, control: np.array([-state[0] + control[0], -state[1]])
        state_dim = 2
        control_dim = 1
        time_step = 0.01
        init_state = np.array([1, -1])
        env = StateSpaceEnv(state_dim, control_dim, ode, time_step, init_state)
        with self.assertRaises(AssertionError):
            StateSpaceEnv(1, control_dim, ode, time_step, init_state)
        with self.assertRaises(TypeError):
            StateSpaceEnv(state_dim, control_dim, lambda t: t, time_step, init_state)
        with self.assertRaises(TypeError):
            StateSpaceEnv(state_dim, control_dim, ode, 0, init_state)
        with self.assertRaises(AssertionError):
            StateSpaceEnv(state_dim, control_dim, ode, time_step, np.array([1, 2, 3]))
        pass

    def test_simulation(self):
        env = self.test_init_correct_input()
        control = np.array([0.])
        state = env._simulation(control)
        ref = np.array([0.99004983, -0.99004983])
        assert(np.all(np.isclose(ref, state)))
        pass

    def test_step(self):
        env = self.test_init_correct_input()
        control = np.array([0.])
        state, cost, done, dummy = env.step(control)
        ref = np.array([0.99004983, -0.99004983])
        assert(np.all(np.isclose(ref, state)))
        assert(not done)
        pass

if __name__ == '__main__':
    unittest.main()