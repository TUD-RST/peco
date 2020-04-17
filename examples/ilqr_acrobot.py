import numpy as np

import matplotlib.pyplot as plt
from probecon.control.ilqr import iLQR
from probecon.system_models.acrobot import Acrobot
from gym.wrappers.monitoring.video_recorder import VideoRecorder

sim_time = 5.

env = Acrobot()

traj_opt = iLQR(env, sim_time, terminal_cost_factor=1000, constrained_state=False)

sol = traj_opt.solve()

env.plot()
plt.savefig('recording/acrobot.pdf')

vid = VideoRecorder(env, 'recording/acrobot.mp4')
env.reset()
for control in sol['controls']:
    env.step(control)
    vid.capture_frame()
vid.close()



