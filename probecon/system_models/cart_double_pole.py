import sympy as sp
import symbtools as st
import pickle
import numpy as np
from numpy import pi, inf
from symbtools import modeltools as mt
from probecon.system_models.core import SymbtoolsEnv, Parameters
from probecon.helpers.symbtools_helpers import create_save_model

class CartDoublePole(SymbtoolsEnv):
    def __init__(self, time_step=0.01, init_state=np.zeros(6),
                 goal_state=None,
                 state_cost=None,
                 control_cost=None,
                 state_bounds=np.array([1.5, 2*pi, 2*pi, inf, inf, inf]),
                 control_bounds=np.array([0.]),
                 mod_file='symbtools_models/cart_double_pole.p',
                 part_lin=True,
                 m0=3.34,
                 m1=0.8512,
                 m2=0.8973,
                 J1=0.01980194,
                 J2=0.02105375,
                 a1=0.20001517,
                 a2=0.26890449,
                 l1=0.32,
                 l2=0.419,
                 d0=0.1,
                 d1=0.00715294,
                 d2=1.9497e-06,
                 g=9.81):

        # parameters:
        self.p = Parameters()
        self.p.m0 = m0
        self.p.m1 = m1
        self.p.m2 = m2
        self.p.J1 = J1
        self.p.J2 = J2
        self.p.a1 = a1
        self.p.a2 = a2
        self.p.l1 = l1
        self.p.l2 = l2
        self.p.d0 = d0
        self.p.d1 = d1
        self.p.d2 = d2
        self.p.g = g

        super(CartDoublePole, self).__init__(mod_file, self.p, time_step, init_state,
                 goal_state=goal_state,
                 state_cost=state_cost,
                 control_cost=control_cost,
                 state_bounds=state_bounds,
                 control_bounds=control_bounds,
                 part_lin=part_lin)


    def render(self, mode='human'):
        screen_width = 800
        world_width = (self.state_space.high[0] + self.p.l1 + self.p.l2)*2
        scale = (screen_width) / world_width
        pole1len = scale * self.p.l1
        pole2len = scale * self.p.l2
        poleslen = pole1len + pole2len
        screen_height = poleslen*2.1
        carty = 0.5 * screen_height  # TOP OF CART
        polewidth = scale * 0.05
        cartwidth = 50.0
        cartheight = 25.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            # add track
            track_left = poleslen - 0.5 * cartwidth
            track_right = screen_width - poleslen + 0.5 * cartwidth
            self.track = rendering.Line((track_left, carty), (track_right, carty))
            self.track.linewidth.stroke = 3
            l, r, t, b = track_left - polewidth, track_left, carty + polewidth, carty - polewidth
            self.track_end_left = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            l, r, t, b = track_right, track_right + polewidth, carty + polewidth, carty - polewidth
            self.track_end_right = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)
            self.viewer.add_geom(self.track_end_left)
            self.viewer.add_geom(self.track_end_right)

            # add cart
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            cart.set_color(.4, .4, .4)
            self.viewer.add_geom(cart)

            # add pole1
            l, r, t, b = -polewidth / 2, polewidth / 2, pole1len - polewidth / 2, -polewidth / 2
            pole1 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole1.set_color(.6, .6, .6)
            self.pole1trans = rendering.Transform(translation=(0, axleoffset))
            pole1.add_attr(self.pole1trans)
            pole1.add_attr(self.carttrans)
            self.viewer.add_geom(pole1)

            # add pole2
            l, r, t, b = -polewidth / 2, polewidth / 2, pole2len - polewidth / 2, -polewidth / 2
            pole2 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole2.set_color(.6, .6, .6)
            self.pole2trans = rendering.Transform(translation=(0, pole1len))
            pole2.add_attr(self.pole2trans)
            pole2.add_attr(self.pole1trans)
            pole2.add_attr(self.carttrans)
            self.viewer.add_geom(pole2)

            # add axle1
            axle1 = rendering.make_circle(1.4 * polewidth / 2)
            axle1.add_attr(self.pole1trans)
            axle1.add_attr(self.carttrans)
            axle1.set_color(.2, .2, .2)
            self.viewer.add_geom(axle1)

            # add axle2
            axle2 = rendering.make_circle(1.4 * polewidth / 2)
            axle2.add_attr(self.pole2trans)
            axle2.add_attr(self.pole1trans)
            axle2.add_attr(self.carttrans)
            axle2.set_color(.2, .2, .2)
            self.viewer.add_geom(axle2)

        if self.state is None: return None

        pos, th1, th2 = self.state[0:3]

        cartx = pos * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.pole1trans.set_rotation(th1)
        self.pole2trans.set_rotation(th2 - th1)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

def modeling():
    params = sp.symbols('m0, m1, m2, J1, J2, a1, a2, l1, l2, d0, d1, d2, g') # system parameters
    m0, m1, m2, J1, J2, a1, a2, l1, l2, d0, d1, d2, g = params

    # force
    F = sp.Symbol('F')

    # generalized coordinates
    qq = sp.Matrix(sp.symbols("q0:3")) # generalized coordinates
    q0, q1, q2 = qq

    # generalized velocities
    dq0, dq1, dq2 = st.time_deriv(qq, qq)

    # generalized velocities
    ddq0, ddq1, ddq2 = st.time_deriv(qq, qq, order=2)

    # position vectors
    p0 = sp.Matrix([q0, 0])
    p1 = p0 + sp.Matrix([-a1*sp.sin(q1), a1*sp.cos(q1)])
    p1_joint = p0 + sp.Matrix([-l1*sp.sin(q1), l1*sp.cos(q1)])
    p2 = p1_joint + sp.Matrix([-a2*sp.sin(q2), a2*sp.cos(q2)])

    # velocity vectors
    dp0 = st.time_deriv(p0, qq)
    dp1 = st.time_deriv(p1, qq)
    dp2 = st.time_deriv(p2, qq)

    # kinetic energy T
    T_rot = (J1*dq1**2 + J2*dq2**2)/2
    T_trans = (m0*dp0.dot(dp0) + m1*dp1.dot(dp1) + m2*dp2.dot(dp2))/2
    T = T_rot + T_trans

    # potential energy V
    V = m1*g*p1[1] + m2*g*p2[1]

    # dissipation function R (Rayleigh dissipation)
    R = (d0*dq0**2 + d1*dq1**2 + d2*(dq2 - dq1)**2)/2

    # external generalized forces
    Q = sp.Matrix([F, 0, 0])

    # Lagrange equations of the second kind
    # d/dt(dL/d(dq_i/dt)) - dL/dq_i + dR/d(dq_i/dt)= Q_
    mod = create_save_model(T, V, qq, Q, R, params, 'symbtools_models/cart_double_pole.p')
    return mod

if __name__ == '__main__':
    modeling()
    init_state = np.array([1.5, -0.5*np.pi, -0.5*np.pi, 0, 0, 0])
    env = CartDoublePole(init_state=init_state)#init_state=np.random.uniform(-1, 1, 8))
    for steps in range(10000):
        state, cost, done, info = env.random_step()
        env.render()
