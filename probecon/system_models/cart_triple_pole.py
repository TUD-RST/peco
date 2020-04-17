import sympy as sp
import symbtools as st
import numpy as np
import pyglet

from numpy import pi, inf

from probecon.system_models.core import SymbtoolsEnv, Parameters
from probecon.helpers.gym_helpers import DrawText
from probecon.helpers.symbtools_helpers import create_save_model

class CartTriplePole(SymbtoolsEnv):
    """
    Class that implements a cart-triple-pole environment

    """
    def __init__(self,
                 time_step=0.002,
                 init_state=np.array([pi, pi, pi, 0., 0., 0., 0., 0.]),
                 goal_state=None,
                 state_cost=None,
                 control_cost=None,
                 state_bounds=np.array([2*pi, 2*pi, 2*pi, 1., inf, inf, inf, inf]),
                 control_bounds=np.array([40.]),
                 mod_file='cart_triple_pole.p',
                 part_lin=True,
                 m0=3.34,
                 m1=0.8512,
                 m2=0.8973,
                 m3=0.5519,
                 J1=0.01980194,
                 J2=0.02105375,
                 J3=0.01818537,
                 a1=0.20001517,
                 a2=0.26890449,
                 a3=0.21666087,
                 l1=0.32,
                 l2=0.419,
                 l3=0.485,
                 d0=0.1,
                 d1=0.00715294,
                 d2=1.9497e-06,
                 d3=0.00164642,
                 g=9.81):
        """

                Args:
                    time_step (float):
                        duration of one time-step
                    init_state:
                        initial state of the environment
                    goal_state (numpy.ndarray):
                        goal state of the environment
                    state_cost (numpy.ndarray):
                        cost of the state vector
                    control_cost (numpy.ndarray):
                        cost of the control vector
                    cost_function (function):
                        explicit cost function (for example a non-quadratic or exponential cost)
                    state_bounds (numpy.ndarray):
                        box constraints of the state space
                    control_bounds (numpy.ndarray):
                        box constraints of the control input space
                    mod_file (string):
                        filename of the pickle file, where the model container was dumped into
                    part_lin (bool):
                        True, if the partial-linearized form of the dynamics should be used
                    m0 (float):
                        mass of the cart in kg
                    m1 (float):
                        mass of the first pole in kg
                    m2 (float):
                        mass of the second pole in kg
                    m3 (float):
                        mass of the third pole in kg
                    J1 (float):
                        rotational moment of inertia of the first pole in kg*m^2
                    J2 (float):
                        rotational moment of inertia of the second pole in kg*m^2
                    J3 (float):
                        rotational moment of inertia of the third pole in kg*m^2
                    a1 (float):
                        position of the center of mass of the first pole in m
                    a2 (float):
                        position of the center of mass of the second pole in m
                    a3 (float):
                        position of the center of mass of the third pole in m
                    l1 (float):
                        length of the first pole in m
                    l2 (float):
                        length of the second pole in m
                    l3 (float):
                        length of the third pole in m
                    d0 (float):
                        damping coefficient of the cart in N*m*s
                    d1 (float):
                        damping coefficient of the first pole in N*m*s
                    d2 (float):
                        damping coefficient of the second pole in N*m*s
                    d3 (float):
                        damping coefficient of the third pole in N*m*s
                    g (float):
                        gravity in m/s^2

                """

        # parameters:
        self.p = Parameters()
        self.p.m0 = m0
        self.p.m1 = m1
        self.p.m2 = m2
        self.p.m3 = m3
        self.p.J1 = J1
        self.p.J2 = J2
        self.p.J3 = J3
        self.p.a1 = a1
        self.p.a2 = a2
        self.p.a3 = a3
        self.p.l1 = l1
        self.p.l2 = l2
        self.p.l3 = l3
        self.p.d0 = d0
        self.p.d1 = d1
        self.p.d2 = d2
        self.p.d3 = d3
        self.p.g = g

        super(CartTriplePole, self).__init__(mod_file, self.p, time_step, init_state,
                                             goal_state=goal_state,
                                             state_cost=state_cost,
                                             control_cost=control_cost,
                                             state_bounds=state_bounds,
                                             control_bounds=control_bounds,
                                             part_lin=part_lin)

    def render(self, mode='human'):
        screen_width = 800
        world_width = (self.state_space.high[3] + self.p.l1 + self.p.l2 + self.p.l3)*2
        scale = (screen_width) / world_width
        pole1len = scale * self.p.l1
        pole2len = scale * self.p.l2
        pole3len = scale * self.p.l3
        poleslen = pole1len + pole2len + pole3len
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
            l, r, t, b = 0.498 * screen_width, .502 * screen_width, carty + 0.5 * polewidth, carty - 0.5 * polewidth
            self.track_middle = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.viewer.add_geom(self.track)
            self.viewer.add_geom(self.track_end_left)
            self.viewer.add_geom(self.track_end_right)
            self.viewer.add_geom(self.track_middle)

            # add cart
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            cart.set_color(.4, .4, .4)
            self.viewer.add_geom(cart)

            # add bar to visualize control input
            l, r, t, b = [0., 0., 0., 0.]
            bar = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            bar.set_color(1., 0., 0.)
            self.bartrans = rendering.Transform(translation=(0., -cartheight / 2))
            bar.add_attr(self.bartrans)
            bar.add_attr(self.carttrans)
            self.viewer.add_geom(bar)

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

            # add time label
            self.label = pyglet.text.Label('',
                                           font_name='Times New Roman',
                                           font_size=12,
                                           x=0.1 * screen_width,
                                           y=0.9 * screen_height,
                                           color=(0, 0, 0, 255))
            self.viewer.add_geom(DrawText(self.label))

        if self.state is None: return None

        time = self.trajectory['time'][-1][0]
        self.label.text = '{0:.2f} s'.format(time)
        th1, th2, th3, pos = self.state[0:4]

        cartx = pos * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.pole1trans.set_rotation(th1)
        self.pole2trans.set_rotation(th2 - th1)
        self.pole3trans.set_rotation(th3 - th2)

        control = self.trajectory['controls'][-1] / self.control_space.high * scale * 0.5
        if control < np.zeros(1):
            l, r, t, b = control, 0., 0., -0.05 * scale
        else:
            l, r, t, b = 0., control, 0., -0.05 * scale
        self.viewer.geoms[5].v = [(l, b), (l, t), (r, t), (r, b)]
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

def modeling():
    params = sp.symbols('m0, m1, m2, m3, J1, J2, J3, a1, a2, a3, l1, l2, l3, d0, d1, d2, d3, g') # system parameters
    m0, m1, m2, m3, J1, J2, J3, a1, a2, a3, l1, l2, l3, d0, d1, d2, d3, g = params

    # force
    F = sp.Symbol('F')

    # generalized coordinates
    qq = sp.Matrix(sp.symbols('q1, q2, q3, q0')) # generalized coordinates
    q1, q2, q3, q0 = qq

    # generalized velocities
    dq1, dq2, dq3, dq0 = st.time_deriv(qq, qq)

    # position vectors
    p0 = sp.Matrix([q0, 0])
    p1 = p0 + sp.Matrix([-a1*sp.sin(q1), a1*sp.cos(q1)])
    p1_joint = p0 + sp.Matrix([-l1*sp.sin(q1), l1*sp.cos(q1)])
    p2 = p1_joint + sp.Matrix([-a2*sp.sin(q2), a2*sp.cos(q2)])
    p2_joint = p1_joint + sp.Matrix([-l2*sp.sin(q2), l2*sp.cos(q2)])
    p3 = p2_joint + sp.Matrix([- a3*sp.sin(q3), a3*sp.cos(q3)])

    # velocity vectors
    dp0 = st.time_deriv(p0, qq)
    dp1 = st.time_deriv(p1, qq)
    dp2 = st.time_deriv(p2, qq)
    dp3 = st.time_deriv(p3, qq)

    # kinetic energy T
    T_rot = (J1*dq1**2 + J2*dq2**2 + J3*dq3**2)*0.5
    T_trans = (m0*dp0.dot(dp0) + m1*dp1.dot(dp1) + m2*dp2.dot(dp2) + m3*dp3.dot(dp3))*0.5
    T = T_rot + T_trans

    # potential energy V
    V = m1*g*p1[1] + m2*g*p2[1] + m3*g*p3[1]

    # dissipation function R (Rayleigh dissipation)
    R = (d0*dq0**2 + d1*dq1**2 + d2*(dq2 - dq1)**2 + d3*(dq3 - dq2)**2)*0.5

    # external generalized forces
    Q = sp.Matrix([F, 0, 0, 0])

    # Lagrange equations of the second kind
    # d/dt(dL/d(dq_i/dt)) - dL/dq_i + dR/d(dq_i/dt)= Q_i
    mod = create_save_model(T, V, qq, Q, R, params, 'symbtools_model_files/cart_triple_pole.p')
    return mod

if __name__ == '__main__':
    modeling()
    init_state = np.array([-0.5 * np.pi, -0.5 * np.pi, -0.5 * np.pi, 0.5, 0, 0, 0, 0])
    env = CartTriplePole(init_state=init_state, time_step=0.02)#init_state=np.random.uniform(-1, 1, 8))
    for steps in range(1000):
        state, cost, done, info = env.random_step()
        env.render()
