
import matplotlib.pyplot as plt
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from probecon.system_models.cart_pole import CartPole
from probecon.control.ilqr import iLQR

env = CartPole()
horizon = 3.
algorithm = iLQR(env, horizon, terminal_cost_factor=100.)
sol = algorithm.solve()
env.plot()
plt.savefig('recording/cartpole.pdf')
vid = VideoRecorder(env, 'recording/cartpole.mp4')
env.reset()
for control in sol['controls']:
    env.step(control)
    vid.capture_frame()
vid.close()