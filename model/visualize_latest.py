import os
import re
import visualization as vi
import utils as ut
import DoomModel as dm
import input as inp
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D

# Next line to silence pyflakes. This import is needed.
Axes3D

FLAGS = tf.app.flags.FLAGS

def print_data(data, fig, subplot, is_3d=True):
  colors = np.arange(0, 180)
  colors = np.concatenate((colors, colors[::-1]))
  colors = vi.duplicate_array(colors, total_length=len(data))

  if is_3d:
    subplot = fig.add_subplot(subplot, projection='3d')
    subplot.set_title('All data')
    subplot.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors, cmap=plt.cm.Spectral, picker=5)
  else:
    subsample = data[0:360] if len(data) < 2000 else data[0:720]
    subsample = np.concatenate((subsample, subsample))[0:len(subsample)+1]
    print(subsample.shape)
    subsample_colors = colors[0:len(subsample)]
    subplot = fig.add_subplot(subplot)
    subplot.set_title('First 360 elem')
    subplot.plot(subsample[:, 0], subsample[:, 1], picker=0)
    subplot.plot(subsample[0, 0], subsample[0, 1], picker=0)
    subplot.scatter(subsample[:, 0], subsample[:, 1], s=50, c=subsample_colors,
                    cmap=plt.cm.Spectral, picker=5)
  return subplot


class EncodingVisualizer:
  def __init__(self, fig, data):
    self.fig = fig
    self.data = data
    self.subplot_3d = print_data(data, fig, 221)
    self.subplot_2d = print_data(data, fig, 223, is_3d=False)
    fig.canvas.mpl_connect('button_press_event', self.on_click)
    fig.canvas.mpl_connect('pick_event', self.on_pick)

    self.model = dm.DoomModel()
    print(FLAGS.input_path)
    self.original_data, _ = inp.get_images(FLAGS.input_path)
    self.reconstructions = self.model.decode(data)

  def on_pick(self, event):
    print(event)
    ind = event.ind
    print(ind)
    print(any([x for x in ind if x < 20]))
    orig = self.original_data[ind]
    reco = self.reconstructions[ind]
    column_picture, height = vi.stitch_images(orig, reco)
    picture = vi.reshape_images(column_picture, height, proportion=2)
    plt.subplot(122).set_title(', '.join(map(str, ind)))
    plt.subplot(122).imshow(picture)
    plt.show()

  def _print_projection(self):
    proj = self.subplot_3d.get_proj()
    print(proj.shape)
    for i, ar in enumerate(proj):
      print(' '.join(map(lambda x: '%+.4f' % x, ar)))
    proj = proj[[0, 1, 3]]
    projected = np.matmul(self.data, proj)
    colors = np.repeat(np.arange(0, 360), int(len(data) / 360) + 1)[0:len(data)]
    colors = colors[0:len(data)]
    self.subplot_proj.clear()
    self.subplot_proj.scatter(projected[:, 0], projected[:, 2], c=colors,
                              cmap=plt.cm.Spectral, picker=5)
    fig.add_subplot(424).clear()
    fig.add_subplot(424).scatter(projected[:, 0], projected[:, 2], c=colors)
    plt.show()

  def on_click(self, event):
    print('click', event)


if __name__ == '__main__':
  FLAGS.load_from_checkpoint = './tmp/doom_bs__act|sigmoid__bs|30__h|500|3|100__init|na__inp|cbd4__lr|0.0004__opt|AO'
  latest_file = ut.get_latest_file(folder=FLAGS.load_from_checkpoint, filter=r'.*\d+\.txt$')
  print(latest_file)
  data = np.loadtxt(latest_file)  # [0:360]

  fig = plt.figure()
  fig.set_size_inches(fig.get_size_inches()[0] * 2, fig.get_size_inches()[1] * 2)
  entity = EncodingVisualizer(fig, data)
  fig.tight_layout()

  plt.show()
