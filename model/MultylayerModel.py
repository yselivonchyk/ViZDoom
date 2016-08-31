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


FLAGS = tf.app.flags.FLAGS


class MultylayerModel(DoomModel.DoomModel):
  model_id = 'ml'

  def __init__(self):
    super(MultylayerModel, self).__init__()

  def encoder(self, input_tensor):
    template = (pt.wrap(input_tensor)
            .flatten()
            .fully_connected(self.layer_encoder)
            .fully_connected(self.layer_encoder)
            .fully_connected(self.layer_narrow))
    return template

  def decoder(self, input_tensor=None, weight_init=tf.truncated_normal):
    return (pt.wrap(input_tensor)
            .fully_connected(self.layer_decoder)
            .fully_connected(self.layer_decoder)
            .fully_connected(self._image_shape[0] * self._image_shape[1] * self._image_shape[2],
                             init=weight_init))

  # def get_meta(self, meta=None):
  #   meta = super(MultylayerModel, self).get_meta()
  #   return meta
  #
  # def load_meta(self, save_path):
  #   meta = super(MultylayerModel, self).get_meta()


if __name__ == '__main__':
  epochs = 100
  if len(sys.argv) > 1:
    epochs = int(sys.argv[1])
  model = MultylayerModel()
  model.set_layer_sizes([500, 12, 500])
  model.train(100)
