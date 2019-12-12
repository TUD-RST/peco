import sympy as sp
import symbtools as st
import numpy as np
from numpy import pi, inf
from symbtools import modeltools as mt

from probecon.system_models.core import StateSpaceEnv, Parameters

class CartTriplePole(StateSpaceEnv):
    def __init__(self, time_step=0.01, init_state=np.zeros(8),
                 goal_state=None,
                 state_cost=None,
                 control_cost=None,
                 state_bounds=np.array([2.4, 2*pi, 2*pi, 2*pi, inf, inf, inf, inf]),
                 control_bounds=np.array([1.])):
        state_dim = 8
        control_dim = 1

        # dummy parameters:
        self.p = Parameters()
        self.p.l1 = .5
        self.p.l2 = .5
        self.p.l3 = .5
        # dummy ODE:
        ode = lambda t, state, control: np.array([state[4], state[5], state[6], state[7], -state[1], -state[2], -state[3], -state[0]])

        super(CartTriplePole, self).__init__(state_dim, control_dim, ode, time_step, init_state,
                 goal_state=goal_state,
                 state_cost=state_cost,
                 control_cost=control_cost,
                 state_bounds=state_bounds,
                 control_bounds=control_bounds)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.state_space.high[0] * 2
        scale = screen_width / world_width
        carty = 200  # TOP OF CART
        polewidth = 10.0
        polelen = scale * self.p.l1
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole1 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole1.set_color(.8, .6, .4)
            self.pole1trans = rendering.Transform(translation=(0, axleoffset))
            pole1.add_attr(self.pole1trans)
            pole1.add_attr(self.carttrans)
            pole2 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole2.set_color(.8, .6, .4)
            self.pole2trans = rendering.Transform(translation=(0, self.p.l2*scale))
            pole2.add_attr(self.pole2trans)
            pole2.add_attr(self.pole1trans)
            pole2.add_attr(self.carttrans)
            pole3 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole3.set_color(.8, .6, .4)
            self.pole3trans = rendering.Transform(translation=(0, self.p.l3*scale))
            pole3.add_attr(self.pole3trans)
            pole3.add_attr(self.pole2trans)
            pole3.add_attr(self.pole1trans)
            pole3.add_attr(self.carttrans)

            self.viewer.add_geom(pole1)
            self.viewer.add_geom(pole2)
            self.viewer.add_geom(pole3)
            self.axle1 = rendering.make_circle(1.2 * polewidth / 2)
            self.axle1.add_attr(self.pole1trans)
            self.axle1.add_attr(self.carttrans)
            self.axle1.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle1)
            self.axle2 = rendering.make_circle(1.2 * polewidth / 2)
            self.axle2.add_attr(self.pole2trans)
            self.axle2.add_attr(self.pole1trans)
            self.axle2.add_attr(self.carttrans)
            self.axle2.set_color(.5, 0., .8)
            self.viewer.add_geom(self.axle2)
            self.axle3 = rendering.make_circle(1.2 * polewidth / 2)
            self.axle3.add_attr(self.pole3trans)
            self.axle3.add_attr(self.pole2trans)
            self.axle3.add_attr(self.pole1trans)
            self.axle3.add_attr(self.carttrans)
            self.axle3.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle3)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        pos, th1, th2, th3 = self.state[0:4]

        cartx = pos * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.pole1trans.set_rotation(th1)
        self.pole2trans.set_rotation(th2 - th1)
        self.pole3trans.set_rotation(th3 - th2 - th1)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

def modeling():
    t = sp.Symbol('t') # time
    params = sp.symbols('m0, m1, m2, m3, J1, J2, J3, a1, a2, a3, l1, l2, l3, g, d0, d1, d2, d3') # system parameters
    m0, m1, m2, m3, J1, J2, J3, a1, a2, a3, l1, l2, l3, g, d0, d1, d2, d3 = params

    # force
    F = sp.Symbol('F')

    # generalized coordinates
    q0_t = sp.Function('q0')(t)
    dq0_t = q0_t.diff(t)
    ddq0_t = q0_t.diff(t, 2)
    q1_t = sp.Function('q1')(t)
    dq1_t = q1_t.diff(t)
    ddq1_t = q1_t.diff(t, 2)
    q2_t = sp.Function('q2')(t)
    dq2_t = q2_t.diff(t)
    ddq2_t = q2_t.diff(t, 2)
    q3_t = sp.Function('q3')(t)
    dq3_t = q3_t.diff(t)
    ddq3_t = q3_t.diff(t, 2)

    # position vectors
    p0 = sp.Matrix([q0_t, 0])
    p1 = sp.Matrix([q0_t - a1*sp.sin(q1_t), a1*sp.cos(q1_t)])
    p2 = sp.Matrix([q0_t - l1*sp.sin(q1_t) - a2*sp.sin(q2_t), l1*sp.cos(q1_t) + a2*sp.cos(q2_t)])
    p3 = sp.Matrix([q0_t - l1*sp.sin(q1_t) - l2*sp.sin(q2_t) - a3*sp.sin(q3_t),
                    l1*sp.cos(q1_t) + l2*sp.cos(q2_t) + a3*sp.cos(q3_t)])

    # velocity vectors
    dp0 = p0.diff(t)
    dp1 = p1.diff(t)
    dp2 = p2.diff(t)
    dp3 = p3.diff(t)

    # kinetic energy T
    T0 = m0/2*(dp0.T*dp0)[0]
    T1 = (m1*(dp1.T*dp1)[0] + J1*dq1_t**2)/2
    T2 = (m2*(dp2.T*dp2)[0] + J2*dq2_t**2)/2
    T3 = (m3*(dp3.T*dp3)[0] + J3*dq3_t**2)/2
    T = T0 + T1 + T2 + T3

    # potential energy V
    V = m1*g*p1[1] + m2*g*p2[1] + m3*g*p3[1]

    # lagrangian L
    L = T - V

    # Lagrange equations of the second kind
    # d/dt(dL/d(dq_i/dt)) - dL/dq_i = Q_i

    Q0 = F - d0*dq0_t
    Q1 =   - d1*dq1_t + d2*(dq2_t - dq1_t)
    Q2 =   - d2*(dq2_t - dq1_t) + d3*(dq3_t - dq2_t)
    Q3 =   - d3*(dq3_t - dq2_t)

    Eq0 = L.diff(dq0_t, t) - L.diff(q0_t) - Q0 # = 0
    Eq1 = L.diff(dq1_t, t) - L.diff(q1_t) - Q1 # = 0
    Eq2 = L.diff(dq2_t, t) - L.diff(q2_t) - Q2 # = 0
    Eq3 = L.diff(dq3_t, t) - L.diff(q3_t) - Q3  # = 0
    # equations of motion
    Eq = sp.Matrix([Eq0, Eq1, Eq2, Eq3])


    # from symbtools.modeltools
    np = 1
    nq = 2
    pp = sp.Matrix(sp.symbols("p1:{0}".format(np+1) ) )
    qq = sp.Matrix(sp.symbols("q1:{0}".format(nq+1) ) )
    ttheta = st.row_stack(pp, qq)
    Q1, Q2 = sp.symbols('Q1, Q2')

    p1_d, q1_d, q2_d = st.time_deriv(ttheta, ttheta)
    p1_dd, q1_dd, q2_dd = st.time_deriv(ttheta, ttheta, order=2)

    p1, q1, q2 = ttheta

    # reordering according to chain
    kk = sp.Matrix([q1, q2, p1])
    kd1, kd2, kd3 = q1_d, q2_d, p1_d
    params = sp.symbols('l1, l2, l3, l4, s1, s2, s3, s4, J1, J2, J3, J4, m1, m2, m3, m4, g')
    l1, l2, l3, l4, s1, s2, s3, s4, J1, J2, J3, J4, m1, m2, m3, m4, g = params

    # geometry
    mey = -sp.Matrix([0,1])

    # coordinates for centers of inertia and joints
    S1 = mt.Rz(kk[0])*mey*s1
    G1 = mt.Rz(kk[0])*mey*l1

    S2 = G1 + mt.Rz(sum(kk[:2]))*mey*s2
    G2 = G1 + mt.Rz(sum(kk[:2]))*mey*l2

    S3 = G2 + mt.Rz(sum(kk[:3]))*mey*s3
    # noinspection PyUnusedLocal
    G3 = G2 + mt.Rz(sum(kk[:3]))*mey*l3

    # velocities of joints and center of inertia
    Sd1 = st.time_deriv(S1, ttheta)
    Sd2 = st.time_deriv(S2, ttheta)
    Sd3 = st.time_deriv(S3, ttheta)

    # energy
    T_rot = ( J1*kd1**2 + J2*(kd1 + kd2)**2 + J3*(kd1 + kd2 + kd3)**2)/2
    T_trans = ( m1*Sd1.T*Sd1 + m2*Sd2.T*Sd2 + m3*Sd3.T*Sd3)/2

    T = T_rot + T_trans[0]
    V = m1*g*S1[1] + m2*g*S2[1] + m3*g*S3[1]

    external_forces = [0, Q1, Q2]
    mod = mt.generate_symbolic_model(T, V, ttheta, external_forces, simplify=False)

    return mod

if __name__ == '__main__':
    # unittest.main()
    env = CartTriplePole(init_state=np.random.uniform(-1, 1, 8))
    for steps in range(10000):
        env.random_step()
        env.render()