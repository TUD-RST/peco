import numpy as np
import time
import cvxopt as opt
import numba
opt.solvers.options['show_progress'] = False

class iLQR(object):
    """
    State- and control-constrained version of the iterative linear-quadratic regulator (iLQR) algorithm,
    that solves nonlinear optimal control problems.

    Implementation based on:

    https://de.mathworks.com/matlabcentral/fileexchange/52069-ilqg-ddp-trajectory-optimization

    Papers:

    1) Y. Tassa, T. Erez, E. Todorov: Synthesis and Stabilization of Complex Behaviours through
    Online Trajectory Optimization
    Link: https://homes.cs.washington.edu/~todorov/papers/TassaIROS12.pdf

    2) Y. Tassa, N. Monsard, E. Todorov: Control-Limited Differential Dynamic Programming
    Link: https://homes.cs.washington.edu/~todorov/papers/TassaICRA14.pdf

    3) J. Chen, W. Zhan, M. Tomizuka: Constrained Iterative LQR for On-Road Autonomous Driving Motion Planning
    Link: https://ieeexplore.ieee.org/abstract/document/8317745

    The basic iLQR algorithm is based on paper 1). To extend the algorithm to incorporate inequality constraints on the control,
    the method in paper 2) solves a quadratic programm (QP).

    To further integrate inequality state constraints (and control constraints) the penalty method from paper 3) is adapted.
    While paper 2) allows for solving the constrained problem directly, paper 3) transforms the constrained problem to
    a non-constrained problem by adding the constraints to the objective-function of the optimization using barrier functions.

    Further reading: https://en.wikipedia.org/wiki/Penalty_method

    We use inequality constraints of the form (x: state, u: control):

        f_1 = (x_k - x_max) <= 0, f_2 = (-x_k + x_min) <= 0, f_3 = (u_k - u_max) <= 0, f_4 = (-u_k + u_min) <= 0

    The barrier function that is used is: b_i = q1*exp(q2*f_i), i = [1, 2, 3, 4]

    Since 'f' is always <=0, if the constraint is not violated, for large q2, this term is close to zero. If however the
    constraint is violated and f>0 the barrier function 'b' takes large values. For details see paper 3).

    To ensure, that the control input is 0. at time step k=0, an exponentially decaying weight 'q3*exp(-q4*k)' with
    parameters 'q3' and 'q4' is added on the control cost.

    """
    def __init__(self, environment, horizon,
                 terminal_cost_factor=0.,
                 max_iterations=500,
                 reg_type=1,
                 tolerance_gradient=1e-4,
                 tolerance_cost=1e-7,
                 parallel_line_search=False,
                 constrained_control=True,
                 constrained_state=True,
                 solve_qp=True,
                 q1=1., # 1.
                 q2=1., # 500.
                 q3=1000.,
                 q4=50.):
        """

        Args:
            environment (probecon.system_model.core.SymbtoolsEnv):
                environment obejct of optimization
            terminal_cost_factor (float):
                scalar, by which the identity matrix is multiplied to form the terminal cost matrix
            max_iterations (int):
                maximum number of iterations
            reg_type(int):
                regularizaton type of the backward pass (see paper: 2)
                    1: Quu + mu*np.eye()
                    2: Vxx + mu*np.eye()
            tolerance_cost (float):
                cost reduction exit criterion
            tolerance_gradient (float):
                gradient exit criterion
            parallel_line_search (bool):
                if True, a parallel line-search is performed
            constrained_control (bool):
                if True, control-constrained version of iLQR is used
            constrained_state (bool):
                if True, state-constrained version of iLQR is used
            solve_qp (bool):
                if True and 'constrained_control==True', the QP
            q1 (float):
                barrier function parameter, gain at the boundary of the constraint
            q2 (float):
                barrier function parameter, parameter in the exponential
            q3 (float):
                exponentially decaying weight parameter, gain of the added cost
            q4 (float):
                exponentially decaying weight parameter, gain of the added cost
            
        """
        self.environment = environment
        if type(horizon) is float:
            self.horizon = int(horizon/self.environment.time_step)
        else:
            self.horizon = horizon
        self.terminal_cost_factor = terminal_cost_factor
        self.terminal_cost_matrix = np.eye(self.environment.state_dim)*self.terminal_cost_factor*self.environment.time_step
        self.max_iterations = max_iterations

        self.tolerance_cost = tolerance_cost
        self.tolerance_gradient = tolerance_gradient

        # regularization parameter (see paper 2) for details)
        self.reg_type = reg_type
        self.mu = 1e-6
        self.mu_max = 1e6
        self.mu_min = 1e-6
        self.mu_init = 1e-6
        self.mu_d = 1.
        self.mu_d0 = 1.6

        # line-search parameters
        self.alphas = 10**np.linspace(0, -3, 11)
        self.parallel = parallel_line_search # parallel forward pass (choose best)

        # if cost - new_cost > zmin, forward pass is successfull
        self.zmin = 0.

        # constraints parameters
        self.constrained_control = constrained_control
        self.constrained_state = constrained_state
        self.solve_qp = solve_qp  # if True, use QP optimization from paper 2), if 2 use augmented cost

        # barrier function parameters (see paper 3) for details)
        self.q1 = q1 # gain at the boundary of the constraint
        self.q2 = q2 # parameter in the exponential

        # exponentially decaying weight parameters
        self.q3 = q3 # gain
        self.q4 = q4 # parameter in the exponential


    def solve(self):
        """
        Solve the trajectory optimziation problem

        Returns:
            sol (dict):
                'states' (numpy.ndarray):
                    (locally) optimal state trajectory, shape: (N, state_dim)
                'controls' (numpy.ndarray):
                    (locally) optimal contorl input trajectory, shape: (N-1, control_dim)
                'K' (numpy.ndarray):
                    feedback matrices, shape: (N-1, state_dim, control_dim)
                'k' (numpy.ndarray):
                    feedforward terms, shape: (N-1, control_dim, )
                'alpha' (float):
                    line-search parameter

        """
        # todo: add an adaptive procedure for tuning the penalty paramters q1 and q2
        start_time = time.time()
        states, controls, cost = self._initial_forward_pass()
        for iter in range(self.max_iterations):
            success_backward = False
            success_forward = False
            # backward pass
            while not success_backward:
                K, k, dV, success_backward = self._backward_pass(states, controls)
                if not success_backward:
                    print('Backward pass not successful')
                    self._increase_mu()
                    if self.mu > self.mu_max:
                        print('mu > mu_max')
                        break

            # check the gradient norm
            g_norm = np.mean(np.max(np.abs(k / (np.abs(controls) + 1)), axis=0))
            if g_norm < self.tolerance_gradient and self.mu < 1e-5:
                self._decrease_mu()
                print('Converged: small gradient')
                break

            states_list = []
            controls_list = []
            cost_list = []
            # forward pass (line-search)
            for alpha in self.alphas:
                new_states, new_controls, new_cost = self._forward_pass(alpha, states, controls, K, k)
                states_list.append(new_states)
                controls_list.append(new_controls)
                cost_list.append(new_cost)
                # check if foward diverged
                if np.any(new_states > 1e8):
                    print('Forward pass diverged.')
                    break

                # check, if forward pass was successful
                cost_red = cost - new_cost
                exp_cost_red = -alpha*(np.sum(dV[:, 0]) + alpha*np.sum(dV[:, 1]))

                if exp_cost_red > 0.:
                    z = cost_red / exp_cost_red
                else:
                    z = np.sign(cost_red)
                    print('Non-positive expected reduction! (Should not occur)')
                if z > self.zmin:
                    success_forward = True
                    if not self.parallel:
                        break

            if success_forward:
                self._decrease_mu()
                best_idx = np.argmin(cost_list)
                states = states_list[best_idx]
                controls = controls_list[best_idx]
                cost = cost_list[best_idx]
                alpha = self.alphas[best_idx]
                print('Iter. {:3} | Cost {:6.5f} | Exp. red. {:6.5f}'.format(iter, cost, exp_cost_red))
                if cost_red < self.tolerance_cost:
                    print('Converged: small cost improvement!')
                    break
            else:
                self._increase_mu()
                print('Iter. {:3} | Forward pass not successful'.format(iter))
                if self.mu > self.mu_max:
                    print('mu > mu_max')
                    break

        sol = {'states': states, 'controls': controls, 'K': K, 'k': k, 'alpha': alpha, 'cost': cost}
        print('Duration: {:3.2f} minutes'.format((time.time()-start_time)/60))
        return sol

    def _forward_pass(self, alpha, states, controls, KK, kk):
        """
        Forward pass

        Returns:
            alpha (float):
                line-search parameter
            states (numpy.ndarray):
                state trajectory, shape: (N, state_dim)
            controls (numpy.ndarray):
                control input trajectory, shape: (N-1, control_dim)
            KK (numpy.ndarray):
                feedback matrices, shape: (N-1, state_dim, control_dim)
            kk (numpy.ndarray):
                feedforward term, shape (N-1, control_dim, )

        Returns:
            states (numpy.ndarray):
                state trajectory, shape: (N, state_dim)
            controls (numpy.ndarray):
                control input trajectory, shape: (N-1, control_dim)
            cost (float):
                cost of the trajectory

        """
        self.environment.reset()
        cost = 0.
        for state, control, K, k in zip(states, controls, KK, kk):
            policy = K@(self.environment.state - state) + alpha*k + control
            if self.constrained_control:
                policy = np.clip(policy, self.environment.control_space.low, self.environment.control_space.high)
            cost += self._constraint_cost(self.environment.state, policy)
            state, reward, done, info = self.environment.step(policy)
            cost -= reward
        states = self.environment.trajectory['states']
        controls = self.environment.trajectory['controls']
        cost += 0.5*states[-1]@self.terminal_cost_matrix@states[-1]
        return states, controls, cost

    def _initial_forward_pass(self):
        """
        Initial forward pass

        Returns:
            states (numpy.ndarray):
                    state trajectory, shape: (N, state_dim)
            controls (numpy.ndarray):
                control input trajectory, shape: (N-1, control_dim)
            cost (float):
                cost of the trajectory

        """
        self.environment.reset()
        cost = 0.
        for k in range(self.horizon):
            control = np.zeros(self.environment.control_dim)
            cost += self._constraint_cost(self.environment.state, control)
            state, reward, done, info = self.environment.step(control)
            cost -= reward
        states = self.environment.trajectory['states']
        controls = self.environment.trajectory['controls']
        cost += 0.5*states[-1]@self.terminal_cost_matrix@states[-1]
        return states, controls, cost

    def _backward_pass(self, states, controls):
        """
        Backward pass

            Args:
                states (numpy.ndarray):
                    state trajectory, shape: (N, state_dim)
                controls (numpy.ndarray):
                    control input trajectory, shape: (N-1, control_dim)

            Returns:
                K (numpy.ndarray):
                    feedback matrices, shape: (N-1, state_dim, control_dim)
                k (numpy.ndarray):
                    feedforward term, shape (N-1, control_dim, )
                dV (numpy.ndarray):
                    estimated cost improvement, shape (N-1, 2, )
                success (bool):
                    False, if any Quu is not positive definite
        """
        A, B = self._taylor_system(states, controls)
        Cxx, Cuu, Cxu, Cux, cx, cu, Cxx_N, cx_N = self._taylor_cost(states, controls)
        Hxx, hx, Huu, hu = self._taylor_constraints(states, controls) # penalty costs
        N = self.horizon + 1

        Vxx = np.zeros((N, self.environment.state_dim, self.environment.state_dim))
        vx = np.zeros((N, self.environment.state_dim))

        K = np.zeros((N-1, self.environment.control_dim, self.environment.state_dim))
        k = np.zeros((N-1, self.environment.control_dim,))

        dV = np.zeros((N-1, 2))

        # boundary condition
        Vxx[N-1] = Cxx_N
        vx[N-1] = cx_N

        success = True

        # compute
        for i in range(N-2, -1, -1): # for i = (N-1,..., 0)
            # quadratic approximation of the Hamiltonian
            Qxx = Cxx[i] + A[i].T@Vxx[i + 1]@A[i] + Hxx[i]
            Quu = Cuu[i] + B[i].T@Vxx[i + 1]@B[i] + Huu[i]
            Qxu = Cxu[i] + A[i].T@Vxx[i + 1]@B[i]
            Qux = Qxu.T
            qx = cx[i] + vx[i + 1]@A[i] + hx[i]
            qu = cu[i] + vx[i + 1]@B[i] + hu[i]

            # regularization
            Vxx_reg = Vxx[i + 1] + self.mu*np.eye(self.environment.state_dim)*(self.reg_type == 2)

            # eq. (10a,10b), paper 1)
            Quu_reg = Cuu[i] + B[i].T@Vxx_reg@B[i] + self.mu*np.eye(self.environment.control_dim)*(self.reg_type == 1)
            Qux_reg = Cux[i] + B[i].T@Vxx_reg@A[i]

            # check if Quu is positive definite
            try:
                np.linalg.cholesky(Quu_reg)
            except np.linalg.LinAlgError as e:
                print('Quu not positve definite!')
                success = False
                break

            if self.solve_qp and self.constrained_control:  # solve QP, eq. (11), paper 2)
                # convert matrices
                Quu_opt = opt.matrix(Quu_reg)
                qu_opt = opt.matrix(qu)

                # inequality constraints Gx <= h, where x is the decision variable
                G = np.kron(np.array([[1.], [-1.]]), np.eye(self.environment.control_dim))
                h = np.array([self.environment.control_space.high - controls[i],
                              -self.environment.control_space.low + controls[i]])
                G_opt = opt.matrix(G)
                h_opt = opt.matrix(h)
                sol = opt.solvers.qp(Quu_opt, qu_opt, G_opt, h_opt)
                k[i] = np.array(sol['x'])

                clamped = np.zeros((self.environment.control_dim), dtype=bool)
                # adapted from boxQP.m of iLQG package
                uplims = np.isclose(k[i], h[:self.environment.control_dim, 0], atol=1e-3)
                lowlims = np.isclose(k[i], h[self.environment.control_dim:, 0], atol=1e-3)

                clamped[uplims] = True
                clamped[lowlims] = True
                free_controls = np.logical_not(clamped)
                if any(free_controls):
                    K[i, free_controls, :] = -np.linalg.solve(Quu_reg, Qux_reg)[free_controls, :]
            else:
                K[i] = -np.linalg.solve(Quu_reg, Qux_reg)  # eq. (10b), paper 1)
                k[i] = -np.linalg.solve(Quu_reg, qu) # eq. (10c), paper 1)

            # cost-to-go approximation
            Vxx[i] = Qxx + K[i].T@Qux + Qxu@K[i] + K[i].T@Quu@K[i]
            Vxx[i] = 0.5*(Vxx[i] + Vxx[i].T) # remain symmetry
            vx[i] = qx + K[i].T@qu + Qxu@k[i] + K[i].T@Quu@k[i]

            dV[i, 0] = qu.T@k[i]
            dV[i, 1] = 0.5*k[i].T@Quu@k[i]

        return K, k, dV, success

    def _taylor_system(self, states, controls):
        """
            1st order Taylor approximation of the dynamics 'f_k(state_k, control_k)'

            Args:
                states (numpy.ndarray):
                    state trajectory, shape: (N, state_dim)
                controls (numpy.ndarray):
                    control input trajectory, shape: (N-1, control_dim)

            Returns:
                A (numpy.ndarray):
                    time-variant system-matrices, shape: (N, state_dim, state_dim)
                B (numpy.ndarray):
                    time_variant input-matrices, shape: (N, state_dim, contorl_dim)

        """
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
        """
        2nd order Taylor approximation of the cost function 'c_k(state_k, control_k)' and 'c_N(state_N)'

        with x_k = state_k - state_ref_k, u_k = control_k - control_ref_k

        index 'ref' indicates trajectory, where the linearization takes place (states, controls)

        for k = 0,...,N-1
        c_k = 0.5*(x_k.T@Cxx[k]@x + u_k.T@Cuu[k]@u_k.T + x_k.T@Cxu[k]@u_k.T + u_k.T@Cux[k]@x_k) + cx_k@x_k + cu_k@u_k)

        c_N = 0.5*x_N.T@Cxx_N@x_N + cx_N@x_N

        Args:
            states (numpy.ndarray):
                state trajectory, shape: (N, state_dim)
            controls (numpy.ndarray):
                control input trajectory, shape: (N-1, control_dim)

        Returns:
            Cxx (numpy.ndarray):
                shape: (N-1, state_dim, state_dim)
            Cuu (numpy.ndarray):
                shape: (N-1, control_dim, control_dim)
            Cxu (numpy.ndarray):
                shape: (N-1, state_dim, control_dim)
            Cux (numpy.ndarray):
                shape: (N-1, control_dim, state_dim)
            cx (numpy.ndarray):
                shape: (N-1, state_dim, )
            cu (numpy.ndarray):
                shape: (N-1, state_dim, )
            Cxx_N (numpy.ndarray):
                shape: (state_dim, state_dim, )
            cx_N (numpy.ndarray):
                shape: (state_dim, )

        """
        if self.environment.cost_function is None:
            S = self.environment.state_cost*self.environment.time_step
            S_N = np.eye(self.environment.state_dim)*self.terminal_cost_factor*self.environment.time_step
            R = self.environment.control_cost*self.environment.time_step
            P = np.zeros((self.environment.state_dim, self.environment.control_dim))

            exp_weight = self.q3*np.exp(-self.q4*np.arange(self.horizon)*self.environment.time_step)
            Cxx = np.kron(S, np.ones((self.horizon, 1, 1)))
            Cuu = np.kron(R, np.ones((self.horizon, 1, 1)) + exp_weight.reshape(self.horizon, 1, 1))
            Cxu = np.kron(P, np.ones((self.horizon, 1, 1)))
            Cux = np.kron(P.T, np.ones((self.horizon, 1, 1)))
            cx = (states[:-1]-self.environment.goal_state)@S + controls@P.T
            cu = (states[:-1]-self.environment.goal_state)@P + controls*Cuu[:, :, 0] # + controls@R
            Cxx_N = S_N
            cx_N = (states[-1]-self.environment.goal_state)@S_N
            return Cxx, Cuu, Cxu, Cux, cx, cu, Cxx_N, cx_N
        else:
            # todo: compute with autodiff from an arbitrary function
            raise NotImplementedError

    def _taylor_constraints(self, states, controls):
        """
        1st and 2nd order Taylor approximation of the barrier function in the penalty method

        See paper 3) section 2.C

        Args:
            states (numpy.ndarray):
                state trajectory, shape: (N, state_dim)
            controls (numpy.ndarray):
                control input trajectory, shape: (N-1, control_dim)

        Returns:
            Hxx (numpy.ndarray):
                hessian of the barrier function with respect to the state, shape: (N, state_dim, state_dim)
            hx (numpy.ndarray):
                jacobian of the barrier function with respect to the state, shape: (N, state_dim, )
            Huu (numpy.ndarray):
            hessian of the barrier function with respect to the control, shape: (N, control_dim, control_dim)
            hu (numpy.ndarray):
                jacobian of the barrier function with respect to the control, shape: (N, control_dim, )
        """
        if self.constrained_state:
            f_high = states[:-1] - self.environment.state_space.high
            f_low = -states[:-1] + self.environment.state_space.low
            hx_high = self.q1*self.q2*np.exp(self.q2*f_high) # equation (11) in paper 3)
            hx_low = -self.q1*self.q2*np.exp(self.q2*f_low) # equation (11) in paper 3)
            hx = hx_high + hx_low
            Hxx = np.array([np.diag(self.q2*(h_high - h_low)) for h_high, h_low in zip(hx_high, hx_low)]) # equation (12) in paper 3)
        else:
            hx = np.zeros((self.horizon, self.environment.state_dim))
            Hxx = np.zeros((self.horizon, self.environment.state_dim, self.environment.state_dim))
        if self.constrained_control and not self.solve_qp:
            f_high = controls - self.environment.control_space.high
            f_low = -controls + self.environment.control_space.low
            hu_high = self.q1 * self.q2 * (np.exp(self.q2 * f_high)) # equation (11) in paper 3)
            hu_low = -self.q1 * self.q2 * (np.exp(self.q2 * f_low)) # equation (11) in paper 3)
            hu = hu_high + hu_low
            Huu = np.array([np.diag(self.q2*(h_high - h_low)) for h_high, h_low in zip(hu_high, hu_low)]) # equation (12) in paper 3)
        else:
            hu = np.zeros((self.horizon, self.environment.control_dim))
            Huu = np.zeros((self.horizon, self.environment.control_dim, self.environment.control_dim))
        return Hxx, hx, Huu, hu

    def _constraint_cost(self, state, control):
        """
        Barrier function for the penalty method (equation (9) in paper 3))

            q1*exp(q2*f)

        Args:
            states (numpy.ndarray):
                state trajectory, shape: (N, state_dim)
            controls (numpy.ndarray):
                control input trajectory, shape: (N-1, control_dim)

        Returns:
            cost (float):
                penalty cost
        """
        cost = 0.
        if self.constrained_state:
            cost += np.sum(self.q1*(np.exp(self.q2*(state - self.environment.state_space.high)) +
                                    np.exp(self.q2*(- state + self.environment.state_space.low))))
        if self.constrained_control and not self.solve_qp:
            cost += np.sum(self.q1*(np.exp(self.q2*(control - self.environment.control_space.high)) +
                                    np.exp(self.q2*(- control + self.environment.control_space.low))))

        return cost

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