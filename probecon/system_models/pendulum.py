import sympy as sp
import symbtools as st
import numpy as np
import pyglet

from numpy import pi, inf

from probecon.system_models.core import SymbtoolsEnv, Parameters
from probecon.helpers.gym_helpers import DrawText
from probecon.helpers.symbtools_helpers import create_save_model

class Pendulum(SymbtoolsEnv):
    """
    Class that implements an inverted pendulum environment

    """
    def __init__(self,
                 time_step=0.02,
                 init_state=np.array([np.pi, 0.]),
                 goal_state=None,
                 state_cost=np.array([1., 0.1]),
                 control_cost=np.array([0.05]),
                 cost_function=None,
                 state_bounds=np.array([2*pi, 2.]),
                 control_bounds=np.array([3.]),
                 mod_file='pendulum.p',
                 ode_error=None,
                 m0=1.0,
                 J0=1.0,
                 l0=1.0,
                 a0=0.5,
                 g=9.81,
                 d0=0.02):
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
             mod_file (str):
                 filename of the pickle file, where the model container was dumped into
             part_lin (bool):
                 True, if the partial-linearized form of the dynamics should be used
             m0 (float):
                 mass of the pole in kg
             J0 (float):
                 rotational moment of inertia of the pole in kg*m^2
             a0 (float):
                 position of the center of mass of the pole in m
             l0 (float):
                 length of the first pole in m
             d0 (float):
                 damping coefficient of the pole in N*m*s
             g (float):
                 gravity in m/s^2
         """

        # parameters:
        self.p = Parameters()
        self.p.m0 = m0
        self.p.J0 = J0
        self.p.a0 = a0
        self.p.l0 = l0
        self.p.d0 = d0
        self.p.g = g

        super(Pendulum, self).__init__(mod_file, self.p, time_step,
                                       init_state=init_state,
                                       goal_state=goal_state,
                                       state_cost=state_cost,
                                       control_cost=control_cost,
                                       cost_function=cost_function,
                                       state_bounds=state_bounds,
                                       control_bounds=control_bounds,
                                       ode_error=ode_error)

    def render(self, mode='human'):
        """
        Renders the inverted pendulum systems current state with OpenGL

        Args:
            mode (string):
                'human':
                    normal render mode
                'rgb_array':
                    render mode for learning from image data

        Returns:
            True if mode='human' and rgb-array if mode='rgb_array'

        """
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

            # add torque arrow
            circle = rendering.make_polyline([])
            self.circle_trans = rendering.Transform(translation=(screen_width / 2, screen_height / 2))
            circle.add_attr(self.circle_trans)
            circle.set_linewidth(5)
            circle.set_color(1., 0.0, 0.0)
            self.viewer.add_geom(circle)

            arrow = rendering.FilledPolygon([])
            self.arrow_trans = rendering.Transform(translation=(screen_width / 2, screen_height / 2))
            arrow.add_attr(self.arrow_trans)
            arrow.set_color(1., 0.0, 0.0)
            self.viewer.add_geom(arrow)

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

        th = self.state[0]
        self.poletrans.set_rotation(th)

        circle = self.viewer.geoms[2]
        arrow = self.viewer.geoms[3]

        # normalized control input
        control = self.trajectory['controls'][-1] / self.control_space.high
        self._arrow_render(arrow, circle, scale, control)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def _arrow_render(self, arrow, circle, scale, input):
        """
        Determines the points of the arrow for rendering.

        Args:
            arrow (gym.envs.classic_control.rendering.FilledPolygon):
                arrow tip
            circle (gym.envs.classic_control.rendering.FilledPolygon):
                circle segment
            scale (int):
                scale of the render determined in self.render()
            input (float):
                normalized control input in the range of [-1., 1.]

        Returns:

        """
        assert (np.abs(input) <= 1.) # check if input is in range [-1., 1.]

        # set radius of the circle
        radius = 0.1*scale

        # determine the segment of the circle
        if input < 0.0:
            min_angle = -np.pi
            max_angle = 0.
        else:
            min_angle = np.pi
            max_angle = 0.

        # scale the segment according to the input
        max_angle = max_angle - (1. - np.abs(input))*(max_angle - min_angle)

        # set the points of the circle segment
        circle.v = [(radius*np.sin(a), radius*np.cos(a)) for a in np.linspace(min_angle, max_angle, 50)]

        # determine the points of the arrow_tip
        arrow_head_in = (0.7*radius * np.sin(max_angle), 0.7*radius*np.cos(max_angle))
        arrow_head_out = (1.3*radius * np.sin(max_angle), 1.3*radius*np.cos(max_angle))
        arrow_head_tip = (1.1*radius*np.sin(max_angle-np.sign(input)*0.5),
                          1.1*radius*np.cos(max_angle-np.sign(input)*0.5))
        # set the points of the arrow tip
        arrow.v = [arrow_head_in, arrow_head_tip, arrow_head_out, arrow_head_in]
        pass

def modeling():
    """
    Derivation of the equations of motion for the inverted pendulum system

    Returns:
        mod (symbtools.modeltools.SymbolicModel):
            contains the equations of motion and other system properties

    """
    params = sp.symbols('m0, J0, a0, l0, d0, g') # system parameters
    m0, J0, a0, l0, d0, g = params

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
    mod = create_save_model(T, V, qq, Q, R, params, 'symbtools_model_files/pendulum.p')
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