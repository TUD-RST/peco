import sympy as sp
import symbtools as st
import numpy as np
import pyglet
import pickle
import matplotlib.pyplot as plt

from numpy import pi, inf
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from peco.system_models.core import SymbtoolsEnv, Parameters
from peco.helpers.gym_helpers import DrawText
from peco.helpers.symbtools_helpers import create_save_model


class CartPole(SymbtoolsEnv):
    """
    Class that implements a cart-pole environment

    """
    def __init__(self, time_step=0.02, init_state=np.array([pi, 0., 0., 0.]),
                 goal_state=None,
                 state_cost=np.array([5., 10., 0.01, 0.01]),
                 control_cost=np.array([0.1]),
                 cost_function=None,
                 state_bounds=np.array([2*pi, 1.2, 10., 4.]),
                 control_bounds=np.array([15.]),
                 mod_file='cart_pole.p',
                 part_lin=True,
                 ode_error=None,
                 m0=3.34,
                 m1=0.3583,
                 J1=0.0379999,
                 l1=0.5,
                 a1=0.43,
                 d0=0.1,
                 d1=0.006588,
                 g=9.81):
        """

        Args:
            time_step (float):
                duration of one time-step
            init_state (numpy.ndarray):
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
            J1 (float):
                rotational moment of inertia of the first pole in kg*m^2
            a1 (float):
                position of the center of mass of the first pole in m
            l1 (float):
                length of the first pole in m
            d0 (float):
                damping coefficient of the cart in N*m*s
            d1 (float):
                damping coefficient of the first pole in N*m*s
            g (float):
                gravity in m/s^2
        """

        # parameters:
        self.p = Parameters()
        self.p.m0 = m0
        self.p.m1 = m1
        self.p.J1 = J1
        self.p.a1 = a1
        self.p.l1 = l1
        self.p.d0 = d0
        self.p.d1 = d1
        self.p.g = g


        super(CartPole, self).__init__(mod_file, self.p, time_step,
                                       init_state=init_state,
                                       goal_state=goal_state,
                                       state_cost=state_cost,
                                       control_cost=control_cost,
                                       cost_function=cost_function,
                                       state_bounds=state_bounds,
                                       control_bounds=control_bounds,
                                       part_lin=part_lin,
                                       ode_error=ode_error)

    def render(self, mode='human'):
        """
        Renders the cart-pole systems current state with OpenGL

        Args:
            mode (string):
                'human':
                    normal render mode
                'rgb_array':
                    render mode for learning from image data

        Returns:
            True if mode='human' and rgb-array if mode='rgb_array'

        """
        screen_width = 800

        world_width = (self.state_space.high[1]+self.p.l1) * 2
        scale = (screen_width) / world_width
        polelen = scale * self.p.l1
        screen_height = 2.5*polelen
        carty = 0.5 * screen_height  # TOP OF CART
        polewidth = scale * 0.04
        cartwidth = 50.0
        cartheight = 25.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            # add track
            track_left = polelen - 0.5 * cartwidth
            track_right = screen_width - polelen + 0.5 * cartwidth
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
            self.bartrans = rendering.Transform(translation=(0., -cartheight/2))
            bar.add_attr(self.bartrans)
            bar.add_attr(self.carttrans)
            self.viewer.add_geom(bar)

            # add pole
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.6, .6, .6)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)

            # add axle
            axle = rendering.make_circle(1.4 * polewidth / 2)
            axle.add_attr(self.poletrans)
            axle.add_attr(self.carttrans)
            axle.set_color(.2, .2, .2)
            self.viewer.add_geom(axle)

            # add time label
            self.label = pyglet.text.Label('',
                                      font_name='Times New Roman',
                                      font_size=12,
                                      x=0.1*screen_width,
                                      y=0.9*screen_height,
                                      color=(0, 0, 0, 255))
            self.viewer.add_geom(DrawText(self.label))

        if self.state is None: return None

        time = self.trajectory['time'][-1][0]
        self.label.text = '{0:.2f} s'.format(time)
        th, pos = self.state[0:2]

        cartx = pos * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(th)

        control = self.trajectory['controls'][-1]/self.control_space.high*scale*0.5
        if control < np.zeros(1):
            l, r, t, b = control, 0., 0., -0.05*scale
        else:
            l, r, t, b = 0., control, 0., -0.05*scale
        self.viewer.geoms[5].v = [(l, b), (l, t), (r, t), (r, b)]
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

def modeling():
    """
    Derivation of the equations of motion for the cart-pole system

    Returns:
        mod (symbtools.modeltools.SymbolicModel):
            contains the equations of motion and other system properties

    """
    params = sp.symbols('m0, m1, J1, a1, l1, d0, d1, g') # system parameters
    m0, m1, J1, a1, l1, d0, d1, g = params

    # force
    F = sp.Symbol('F')

    # generalized coordinates
    qq = sp.Matrix(sp.symbols('q1, q0')) # generalized coordinates
    q1, q0 = qq

    # generalized velocities
    dq1, dq0 = st.time_deriv(qq, qq)

    # position vectors
    p0 = sp.Matrix([q0, 0])
    p1 = p0 + sp.Matrix([-a1*sp.sin(q1), a1*sp.cos(q1)])

    # velocity vectors
    dp0 = st.time_deriv(p0, qq)
    dp1 = st.time_deriv(p1, qq)

    # kinetic energy T
    T_rot = (J1*dq1**2)*0.5
    T_trans = (m0*dp0.dot(dp0) + m1*dp1.dot(dp1))*0.5
    T = T_rot + T_trans

    # potential energy V
    V = m1*g*p1[1]

    # dissipation function R (Rayleigh dissipation)
    R = (d0*dq0**2 + d1*dq1**2)*0.5

    # generalized forces
    Q = sp.Matrix([0, F])

    # Lagrange equations of the second kind
    # d/dt(dL/d(dq_i/dt)) - dL/dq_i + dR/d(dq_i/dt)= Q_i
    mod = create_save_model(T, V, qq, Q, R, params, 'symbtools_model_files/cart_pole.p')
    return mod

if __name__ == '__main__':
    modeling()
    init_state = np.array([-0.5*np.pi, 0.5, 0., 0.])
    env = CartPole(init_state=init_state)
    vid = VideoRecorder(env, 'recording/video.mp4')
    env.reset()
    for i in range(500):
        env.random_step()
        #env.render()
        vid.capture_frame()
    vid.close()
    #env.close()