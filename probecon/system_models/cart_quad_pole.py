import sympy as sp
import pickle
import symbtools as st
import numpy as np
import pyglet

from numpy import pi, inf
from symbtools import modeltools as mt
from probecon.system_models.core import SymbtoolsEnv, Parameters
from probecon.helpers.gym_helpers import DrawText
from probecon.helpers.symbtools_helpers import create_save_model

class CartQuadPole(SymbtoolsEnv):
    def __init__(self, time_step=0.0125, init_state=np.zeros(6),
                 goal_state=None,
                 state_cost=None,
                 control_cost=None,
                 state_bounds=np.array([2*pi, 2*pi, 2*pi, 2*pi, 1.5, inf, inf, inf, inf, inf]),
                 control_bounds=np.array([0.]),
                 mod_file='symbtools_models/cart_quad_pole.p',
                 part_lin=True,
                 m0=3.34,
                 m1=0.8512,
                 m2=0.8973,
                 m3=0.5519,
                 m4=0.5519,
                 J1=0.01980194,
                 J2=0.02105375,
                 J3=0.01818537,
                 J4=0.0179,
                 a1=0.20001517,
                 a2=0.26890449,
                 a3=0.21666087,
                 a4=0.213,
                 l1=0.32,
                 l2=0.419,
                 l3=0.485,
                 l4=0.485,
                 d0=0.1,
                 d1=0.00715294,
                 d2=1.9497e-06,
                 d3=0.00164642,
                 d4=0.0017,
                 g=9.81):

        # parameters:
        self.p = Parameters()
        self.p.m0 = m0
        self.p.m1 = m1
        self.p.m2 = m2
        self.p.m3 = m3
        self.p.m4 = m4
        self.p.J1 = J1
        self.p.J2 = J2
        self.p.J3 = J3
        self.p.J4 = J4
        self.p.a1 = a1
        self.p.a2 = a2
        self.p.a3 = a3
        self.p.a4 = a4
        self.p.l1 = l1
        self.p.l2 = l2
        self.p.l3 = l3
        self.p.l4 = l4
        self.p.d0 = d0
        self.p.d1 = d1
        self.p.d2 = d2
        self.p.d3 = d3
        self.p.d4 = d4
        self.p.g = g

        super(CartQuadPole, self).__init__(mod_file, self.p, time_step, init_state,
                                           goal_state=goal_state,
                                           state_cost=state_cost,
                                           control_cost=control_cost,
                                           state_bounds=state_bounds,
                                           control_bounds=control_bounds,
                                           part_lin=part_lin)

    def render(self, mode='human'):
        screen_width = 800
        world_width = (self.state_space.high[4] + self.p.l1 + self.p.l2 + self.p.l3 + self.p.l4)*2
        scale = (screen_width) / world_width
        pole1len = scale * self.p.l1
        pole2len = scale * self.p.l2
        pole3len = scale * self.p.l3
        pole4len = scale * self.p.l4
        poleslen = pole1len + pole2len + pole3len + pole4len
        screen_height = 2.1*poleslen
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

            # add pole3
            l, r, t, b = -polewidth / 2, polewidth / 2, pole3len - polewidth / 2, -polewidth / 2
            pole3 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole3.set_color(.6, .6, .6)
            self.pole3trans = rendering.Transform(translation=(0, pole2len))
            pole3.add_attr(self.pole3trans)
            pole3.add_attr(self.pole2trans)
            pole3.add_attr(self.pole1trans)
            pole3.add_attr(self.carttrans)
            self.viewer.add_geom(pole3)

            # add pole4
            l, r, t, b = -polewidth / 2, polewidth / 2, pole4len - polewidth / 2, -polewidth / 2
            pole4 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole4.set_color(.6, .6, .6)
            self.pole4trans = rendering.Transform(translation=(0, pole3len))
            pole4.add_attr(self.pole4trans)
            pole4.add_attr(self.pole3trans)
            pole4.add_attr(self.pole2trans)
            pole4.add_attr(self.pole1trans)
            pole4.add_attr(self.carttrans)
            self.viewer.add_geom(pole4)

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

            # add axle3
            axle3 = rendering.make_circle(1.4 * polewidth / 2)
            axle3.add_attr(self.pole3trans)
            axle3.add_attr(self.pole2trans)
            axle3.add_attr(self.pole1trans)
            axle3.add_attr(self.carttrans)
            axle3.set_color(.2, .2, .2)
            self.viewer.add_geom(axle3)

            # add axle4
            axle4 = rendering.make_circle(1.4 * polewidth / 2)
            axle4.add_attr(self.pole4trans)
            axle4.add_attr(self.pole3trans)
            axle4.add_attr(self.pole2trans)
            axle4.add_attr(self.pole1trans)
            axle4.add_attr(self.carttrans)
            axle4.set_color(.2, .2, .2)
            self.viewer.add_geom(axle4)

            # add time label
            self.label = pyglet.text.Label('',
                                           font_name='Times New Roman',
                                           font_size=12,
                                           x=0.1 * screen_width,
                                           y=0.9 * screen_height,
                                           color=(0, 0, 0, 255))
            self.viewer.add_geom(DrawText(self.label))

        if self.state is None: return None

        time = self.trajectory['time'][-1]
        self.label.text = '{0:.2f} s'.format(time, '2f')
        th1, th2, th3, th4, pos = self.state[0:5]

        cartx = pos * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.pole1trans.set_rotation(th1)
        self.pole2trans.set_rotation(th2 - th1)
        self.pole3trans.set_rotation(th3 - th2)
        self.pole4trans.set_rotation(th4 - th3)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

def modeling():
    params = sp.symbols('m0, m1, m2, m3, m4, J1, J2, J3, J4, a1, a2, a3, a4, l1, l2, l3, l4, d0, d1, d2, d3, d4, g') # system parameters
    m0, m1, m2, m3, m4, J1, J2, J3, J4, a1, a2, a3, a4, l1, l2, l3, l4, d0, d1, d2, d3, d4, g = params

    # force
    F = sp.Symbol('F')

    # generalized coordinates
    qq = sp.Matrix(sp.symbols('q1, q2, q3, q4, q0')) # generalized coordinates
    q1, q2, q3, q4, q0 = qq

    # generalized velocities
    dq1, dq2, dq3, dq4, dq0 = st.time_deriv(qq, qq)

    # position vectors
    p0 = sp.Matrix([q0, 0])
    p1 = p0 + sp.Matrix([-a1*sp.sin(q1), a1*sp.cos(q1)])
    p1_joint = p0 + sp.Matrix([-l1*sp.sin(q1), l1*sp.cos(q1)])
    p2 = p1_joint + sp.Matrix([-a2*sp.sin(q2), a2*sp.cos(q2)])
    p2_joint = p1_joint + sp.Matrix([-l2*sp.sin(q2), l2*sp.cos(q2)])
    p3 = p2_joint + sp.Matrix([- a3*sp.sin(q3), a3*sp.cos(q3)])
    p3_joint = p2_joint + sp.Matrix([-l3*sp.sin(q3), l3*sp.cos(q3)])
    p4 = p3_joint + sp.Matrix([-a4*sp.sin(q4), a4*sp.cos(q4)])

    # velocity vectors
    dp0 = st.time_deriv(p0, qq)
    dp1 = st.time_deriv(p1, qq)
    dp2 = st.time_deriv(p2, qq)
    dp3 = st.time_deriv(p3, qq)
    dp4 = st.time_deriv(p4, qq)

    # kinetic energy T
    T_rot = (J1*dq1**2 + J2*dq2**2 + J3*dq3**2 + J4*dq4**2)/2
    T_trans = (m0*dp0.dot(dp0) + m1*dp1.dot(dp1) + m2*dp2.dot(dp2) + m3*dp3.dot(dp3) + m4*dp4.dot(dp4))/2
    T = T_rot + T_trans

    # potential energy V
    V = m1*g*p1[1] + m2*g*p2[1] + m3*g*p3[1] + m4*g*p4[1]

    # dissipation function R (Rayleigh dissipation)
    R = (d0*dq0**2 + d1*dq1**2 + d2*(dq2 - dq1)**2 + d3*(dq3 - dq2)**2 + d4*(dq4 - dq3)**2)/2

    # external generalized forces
    Q = sp.Matrix([0, 0, 0, 0, F])

    # Lagrange equations of the second kind
    # d/dt(dL/d(dq_i/dt)) - dL/dq_i + dR/d(dq_i/dt)= Q_i
    mod = create_save_model(T, V, qq, Q, R, params, 'symbtools_models/cart_quad_pole.p')
    return mod

if __name__ == '__main__':
    #modeling()
    # unittest.main()
    init_state = np.array([-0.5 * np.pi, -0.5 * np.pi, -0.5 * np.pi,-0.5 * np.pi, 1.5, 0, 0, 0, 0, 0])
    env = CartQuadPole(init_state=init_state)#init_state=np.random.uniform(-1, 1, 8))
    for steps in range(100):
        state, cost, done, info = env.random_step()
        env.render()
