import numpy as np

import matplotlib.pyplot as plt
from probecon.control.ilqr import iLQR
from probecon.system_models.acrobot import Acrobot
from gym.wrappers.monitoring.video_recorder import VideoRecorder

sim_time = 5.

env = Acrobot()

traj_opt = iLQR(env, sim_time, terminal_cost_factor=100)

sol = traj_opt.solve()

vid = VideoRecorder(env, 'recording/video.mp4')

for control in sol['controls']:
    env.step(control)
    vid.capture_frame()
vid.close()
env.plot()
plt.show()


