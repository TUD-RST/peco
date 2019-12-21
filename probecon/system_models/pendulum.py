import sympy as sp
import symbtools as st
import numpy as np
import pyglet

from numpy import pi, inf

from probecon.system_models.core import SymbtoolsEnv, Parameters
from probecon.helpers.gym_helpers import DrawText
from probecon.helpers.symbtools_helpers import create_save_model

class Pendulum(SymbtoolsEnv):
    def __init__(self, time_step=0.01, init_state=np.zeros(2),
                 goal_state=None,
                 state_cost=None,
                 control_cost=None,
                 state_bounds=np.array([2*pi, inf]),
                 control_bounds=np.array([1.]),
                 mod_file='pendulum.p',
                 m0=0.3583,
                 J0=0.0379999,
                 l0=0.5,
                 a0=0.43,
                 g=9.81,
                 d0=0.006588):

        # parameters:
        self.p = Parameters()
        self.p.m0 = m0
        self.p.J0 = J0
        self.p.a0 = a0
        self.p.l0 = l0
        self.p.g = g
        self.p.d0 = d0

        super(Pendulum, self).__init__(mod_file, self.p, time_step, init_state,
                 goal_state=goal_state,
                 state_cost=state_cost,
                 control_cost=control_cost,
                 state_bounds=state_bounds,
                 control_bounds=control_bounds)

    def render(self, mode='human'):
        screen_width = 400

        world_width = 2.5*self.p.l0
        scale = (screen_width) / world_width
        polelen = scale * self.p.l0
        screen_height = 2.5*polelen
        axley = 0.5 * screen_height  # TOP OF CART
        polewidth = scale*0.04

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # add pole
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.6, .6, .6)
            self.poletrans = rendering.Transform(translation=(0.5*screen_width, axley))
            pole.add_attr(self.poletrans)
            self.viewer.add_geom(pole)

            # add axle
            axle = rendering.make_circle(1.4 * polewidth / 2)
            axle.add_attr(self.poletrans)
            axle.set_color(.2, .2, .2)
            self.viewer.add_geom(axle)

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
        self.label.text = '{0:.2f} s'.format(time)

        th = self.state[0]
        self.poletrans.set_rotation(th)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')



def modeling():
    params = sp.symbols('m0, J0, a0, l0, g, d0') # system parameters
    m0, J0, a0, l0, g, d0 = params

    # force
    tau = sp.Symbol('tau')

    # generalized coordinates
    qq = sp.Matrix(sp.symbols("q0:1")) # generalized coordinates
    q0, = qq

    # generalized velocities
    dq0, = st.time_deriv(qq, qq)

    # position vectors
    p0 = sp.Matrix([-a0*sp.sin(q0), a0*sp.cos(q0)])

    # velocity vectors
    dp0 = st.time_deriv(p0, qq)

    # kinetic energy T
    T_rot = (J0*dq0**2)*0.5
    T_trans = (m0*dp0.dot(dp0))*0.5
    T = T_rot + T_trans

    # potential energy V
    V = m0*g*p0[1]

    # dissipation function R (Rayleigh dissipation)
    R = (d0*dq0**2)*0.5

    # external generalized forces
    Q = sp.Matrix([tau])

    # Lagrange equations of the second kind
    # d/dt(dL/d(dq_i/dt)) - dL/dq_i + dR/d(dq_i/dt)= Q_i
    mod = create_save_model(T, V, qq, Q, R, params, 'symbtools_models/cart_pole.p')
    return mod

if __name__ == '__main__':
    modeling()
    env = Pendulum()
    for i in range(1000):
        env.random_step()
        if i == 500:
            env.p.d0=10000
            env.params_vals = list(env.p.__dict__.values())
        env.render()