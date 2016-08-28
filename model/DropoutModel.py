"""Doom AE with dropout. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import json, os, re
import numpy as np
import utils as ut
import input as inp
import tools.checkpoint_utils as ch_utils
import activation_functions as act
import visualization as vis
import prettytensor as pt
import sys
import prettytensor.bookkeeper as bookkeeper
from prettytensor.tutorial import data_utils
import DoomModel


tf.app.flags.DEFINE_float('dropout', 0.9, 'Dropout rate for pre-hidden layer features')


FLAGS = tf.app.flags.FLAGS


class DropoutModel(DoomModel.DoomModel):
  def __init__(self):
    FLAGS.suffix = 'doom_do'
    super(DropoutModel, self).__init__()

  def encoder(self, input_tensor):
    print('Dropout encoder, dropout rate: %f' % FLAGS.dropout)
    template = (pt.wrap(input_tensor)
            .flatten()
            .fully_connected(self.layer_encoder)
            .dropout(FLAGS.dropout)
            .fully_connected(self.layer_narrow))
    return template

  def get_meta(self, meta=None):
    meta = super(DropoutModel, self).get_meta()
    meta['do'] = FLAGS.dropout
    return meta

  def load_meta(self, save_path):
    meta = super(DropoutModel, self).get_meta()
    FLAGS.dropout = float(meta['do'])


if __name__ == '__main__':
  model = DropoutModel()
  model.set_layer_sizes([500, 12, 500])
  model.train(100)
