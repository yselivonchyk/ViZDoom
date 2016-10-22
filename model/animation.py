import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['animation.ffmpeg_path'] = '/opt/local/bin/ffmpeg'

fig = plt.figure()
ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
line, = ax.plot([], [], lw=2)


def init():
    line.set_data([], [])
    return line,

x = np.linspace(0, 2, 1000)

def animate(i):
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    line.set_data(x, y)
    return line,

from matplotlib import animation

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=20, blit=True)
anim.save('basic_animation.mp4', fps=30)