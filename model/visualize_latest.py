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
    ut.print_info('subsample shape %s' % str(subsample.shape))
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
    self.data = data
    self.fig = fig
    vi.visualize_encodings(data, grid=(3, 5), skip_every=5, fast=fast, fig=fig, interactive=True)
    plt.subplot(155).set_title(', '.join('hold on'))
    fig.canvas.mpl_connect('button_press_event', self.on_click)
    fig.canvas.mpl_connect('pick_event', self.on_pick)
    try:
      ut.print_info('Checkpoint: %s' % FLAGS.load_from_checkpoint)
      self.model = dm.DoomModel()
      self.reconstructions = self.model.decode(data)
    except:
      ut.print_info("Model could not load from checkpoint", color=31)
      ut.print_info('INPUT: %s' % FLAGS.input_path.split('/')[-3])
      self.original_data, _ = inp.get_images(FLAGS.input_path)
      self.reconstructions = np.zeros(self.original_data.shape).astype(np.uint8)

  def on_pick(self, event):
    print(event)
    ind = event.ind
    print(ind)
    print(any([x for x in ind if x < 20]))
    orig = self.original_data[ind]
    reco = self.reconstructions[ind]
    column_picture, height = vi.stitch_images(orig, reco)
    picture = vi.reshape_images(column_picture, height, proportion=3)
    plt.subplot(155).set_title(', '.join(map(str, ind)))
    plt.subplot(155).imshow(picture)
    plt.show()

  def on_click(self, event):
    print('click', event)


def visualize_latest_from_visualization_folder():
  latest_file = ut.get_latest_file(filter=r'.*\d+\.txt$')
  ut.print_info('Encoding file: %s' % latest_file.split('/')[-1])
  data = np.loadtxt(latest_file)  # [0:360]
  fig = plt.figure()
  vi.visualize_encodings(data, fast=fast, fig=fig,  interactive=True)
  fig.suptitle(latest_file.split('/')[-1])
  fig.tight_layout()
  plt.show()


def visualize_from_checkpoint(checkpoint, epoch=None):
  FLAGS.load_from_checkpoint = checkpoint
  file_filter = r'.*\d+\.txt$' if epoch is None else r'.*e\|%d.*' % epoch
  latest_file = ut.get_latest_file(folder=FLAGS.load_from_checkpoint, filter=file_filter)
  ut.print_info('Encoding file: %s' % latest_file.split('/')[-1])
  data = np.loadtxt(latest_file)
  fig = plt.figure()
  fig.set_size_inches(fig.get_size_inches()[0] * 2, fig.get_size_inches()[1] * 2)
  entity = EncodingVisualizer(fig, data)
  # fig.tight_layout()
  plt.show()


fast = True
if __name__ == '__main__':
  visualize_latest_from_visualization_folder()
  # visualize_from_checkpoint(
  #   checkpoint='./tmp/doom_bs__act|sigmoid__bs|20__h|500|5|500__init|na__inp' \
  #                             '|cbd4__lr|0.0004__opt|AO',
  #   epoch=250
  # )