import numpy as np
from probecon.control.trajectory_optimization import TrajectoryOptimization
from probecon.system_models.acrobot import Acrobot
from gym.wrappers.monitoring.video_recorder import VideoRecorder

import matplotlib.pyplot as plt

def c_k(x, u):
    x2, x1, x4, x3 = x
    u1, = u
    c = 2 * x1 ** 2 + 0.5 * x2 ** 2 + 0.02 * x3 ** 2 + 0.05 * x4 ** 2 + 0.01 * u1 ** 2
    return c


def c_N(x):
    x2, x1, x4, x3 = x
    c = 100 * x1 ** 2 + 100 * x2 ** 2 + 100 * x3 ** 2 + 100 * x4 ** 2
    return c


sim_time = 5.

env = Acrobot(cost_function=c_k)

traj_opt = TrajectoryOptimization(env, sim_time, terminal_cost=c_N)

sol = traj_opt.solve()

vid = VideoRecorder(env, 'recording/video.mp4')

for control in sol['u_sim']:
    env.step(control)
    vid.capture_frame()
vid.close()
env.plot()
plt.show()


