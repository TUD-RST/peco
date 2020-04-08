import torch
import numpy as np
import matplotlib.pyplot as plt
from probecon.system_models.pendulum import Pendulum

class iLQR(object):
    """
    Class that wraps different trajectory optimization solvers.

    """
    def __init__(self, environment, horizon):
        """

        Args:
            
        """
        self.environment = environment
        self.horizon = horizon
        self.terminal_cost_factor = 100.
        self.max_iterations = 500
        self.zmin = 0.
        self.tolerance_cost = 1e-7
        self.tolerance_gradient = 1e-4
        # regularization parameter
        self.mu = 1e-6
        self.mu_max = 1e6
        self.mu_min = 1e-6
        self.mu_init = 1e-6
        self.mu_d = 1.
        self.mu_d0 = 1.6
        self.alphas = 10**np.linspace(0, -3, 11) # line-search parameters
        self.parallel = True # parallel forward pass (choose best)

    def solve(self):
        # initial forward pass
        states, controls, cost = self._initial_forward_pass()
        for iter in range(self.max_iterations):
            success_backward = False
            success_forward = False
            success_gradient = False

            # backward pass
            while not success_backward:
                K, k, dV, success_backward = self._backward_pass(states, controls)
                if not success_backward:
                    print('Backward not successful')
                    self._increase_mu()
                    break

            # check the gradient norm
            g_norm = np.mean(np.max(np.abs(k / (np.abs(controls) + 1)), axis=0))
            if g_norm < self.tolerance_gradient and self.mu < 1e-5:
                self._decrease_mu()
                success_gradient = True

            # forward pass (line-search)
            for alpha in self.alphas:
                new_states, new_controls, new_cost = self._forward_pass(alpha, states, controls, K, k)

                # check, if forward pass was successful
                cost_red = cost - new_cost
                exp_cost_red = -alpha*(np.sum(dV[:, 0]) + alpha*np.sum(dV[:, 1]))

                if exp_cost_red > 0.:
                    z = cost_red / exp_cost_red
                else:
                    z = np.sign(cost_red)
                    print('non-positive expected reduction')
                    # self.increase_mu() # todo: probably delete this line, if something's not working!
                if z > self.zmin:
                    success_forward = True
                    break

            if success_forward:
                self._decrease_mu()

                states = new_states
                controls = new_controls
                cost = new_cost
                print('iter {}, cost {:6.5f}'.format(iter, cost))

                if cost_red < self.tolerance_cost:
                    print('Converged: small cost improvement')
                    break

                if success_gradient:
                    print('Converged: small gradient')
                    break

            else:
                self._increase_mu()
                print('Forward not successfull, mu {}'.format(self.mu))
                if self.mu > self.mu_max:
                    print('Diverged: no improvement')
                    break

        sol = {'states': states, 'controls': controls, 'K': K, 'k': k, 'alpha': alpha, 'cost': cost}
        return sol

    def _forward_pass(self, alpha, states, controls, KK, kk):
        self.environment.reset()
        cost = 0.
        for state, control, K, k in zip(states, controls, KK, kk):
            policy = K@(env.state - state) + alpha*k + control
            state, reward, done, info = self.environment.step(policy)
            cost -= reward
        states = self.environment.trajectory['states']
        controls = self.environment.trajectory['controls']
        return states, controls, cost

    def _initial_forward_pass(self):
        self.environment.reset()
        cost = 0.
        for k in range(self.horizon):
            control = np.zeros(self.environment.control_dim)
            state, reward, done, info = self.environment.step(control)
            cost -= reward
        states = self.environment.trajectory['states']
        controls = self.environment.trajectory['controls']
        return states, controls, cost

    def _backward_pass(self, states, controls):
        A, B = self._taylor_system(states, controls)
        Cxx, Cuu, Cxu, Cux, cx, cu, Cxx_N, cx_N = self._taylor_cost(states, controls)
        N = self.horizon

        Vxx = np.zeros((N + 1, self.environment.state_dim, self.environment.state_dim))
        vx = np.zeros((N + 1, self.environment.state_dim))

        K = np.zeros((N, self.environment.control_dim, self.environment.state_dim))
        k = np.zeros((N, self.environment.control_dim,))

        dV = np.zeros((N, 2))

        # boundary condition
        Vxx[N] = Cxx_N
        vx[N] = cx_N

        success = True

        # compute
        for i in range(N-1, -1, -1): # for i = (N-1,..., 0)
            # quadratic approximation of the Hamiltonian
            Qxx = Cxx[i] + A[i].T@Vxx[i+1]@A[i]
            Quu = Cuu[i] + B[i].T@Vxx[i+1]@B[i]
            Qxu = Cxu[i] + A[i].T@Vxx[i+1]@B[i]
            Qux = Qxu.T
            qx = cx[i] + vx[i+1]@A[i]
            qu = cu[i] + vx[i+1]@B[i]

            # check if Quu is positive definite
            try:
                np.linalg.cholesky(Quu)
            except np.linalg.LinAlgError as e:
                print(e)
                success = False
                break

            Quu_inv = Quu**-1

            #
            # controller gains
            K[i] = -Quu_inv@Qux
            k[i] = -Quu_inv@qu

            # cost-to-go approximation
            Vxx[i] = Qxx + K[i].T@Qux + Qxu@K[i] + K[i].T@Quu@K[i]
            Vxx[i] = 0.5*(Vxx[i] + Vxx[i].T) # remain symmetry
            vx[i] = qx + K[i].T@qu + Qxu@k[i] + K[i].T@Quu@k[i]

            dV[i, 0] = qu.T@k[i]
            dV[i, 1] = 0.5*k[i].T@Quu@k[i]

        return K, k, dV, success

    def _taylor_system(self, states, controls):
        if self.environment.__class__.__bases__[0].__name__=='SymbtoolsEnv':
            A = np.array([np.eye(self.environment.state_dim)
                          + self.environment.time_step*self.environment.ode_state_jac(state, control)
                          for state, control in zip(states, controls)])
            B = np.array([self.environment.time_step*self.environment.ode_control_jac(state, control)
                          for state, control in zip(states, controls)])
        else:
            # todo: finite diff as fallback
            # todo: what if ode_error != None
            # todo: what if ode_error = nn.Module -> autodiff
            # todo: what if env = StateSpaceModel Deep Ensemble?
            raise NotImplementedError
        return A, B

    def _taylor_cost(self, states, controls):
        # states with shape (N+1, state_dim)
        # controls with shape (N, control_dim)
        if self.environment.cost_function is None:
            S = self.environment.state_cost*self.environment.time_step
            S_N = S * self.terminal_cost_factor*self.environment.time_step
            R = self.environment.control_cost*self.environment.time_step
            P = np.zeros((self.environment.state_dim, self.environment.control_dim))

            Cxx = np.kron(S, np.ones((self.horizon, 1, 1))) # shape (N, state_dim, state_dim)
            Cuu = np.kron(R, np.ones((self.horizon, 1, 1)))  # shape (N, control_dim, control_dim)
            Cxu = np.kron(P, np.ones((self.horizon, 1, 1))) # shape (N, state_dim, control_dim)
            Cux = np.kron(P.T, np.ones((self.horizon, 1, 1))) # shape (N, control_dim, state_dim)
            cx = np.array([state@S + control@P.T for (state, control) in zip(states, controls)]) # shape (N, state_dim)
            cu = np.array([state@P + control@R for (state, control) in zip(states, controls)]) # shape (N, control_dim)
            Cxx_N = S_N # shape (state_dim, state_dim)
            cx_N = states[-1]@S_N # shape (state_dim, )
            return Cxx, Cuu, Cxu, Cux, cx, cu, Cxx_N, cx_N
        else:
            # todo: compute with autodiff from an arbitrary function
            raise NotImplementedError

    def _decrease_mu(self):
        """
        Decrease regularization parameter mu. Section F, paper 1)
        """
        self.mu_d = min(self.mu_d / self.mu_d0, 1 / self.mu_d0)
        self.mu = self.mu * self.mu_d * (self.mu > self.mu_min)
        pass

    def _increase_mu(self):
        """
        Increase regularization parameter mu. Section F, paper 1)
        """
        self.mu_d = max(self.mu_d0, self.mu_d0 * self.mu_d)
        self.mu = max(self.mu_min, self.mu * self.mu_d)
        pass

if __name__=="__main__":
    env = Pendulum()
    horizon = 400
    algo = iLQR(env, horizon)
    algo.solve()
    env.plot()
    plt.show()

