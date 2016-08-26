'''TensorFlow implementation of http://arxiv.org/pdf/1312.6114v10.pdf'''

from __future__ import absolute_import, division, print_function

import math
import os

import numpy as np
import prettytensor as pt
import scipy.misc
import tensorflow as tf
from scipy.misc import imsave
from tensorflow.examples.tutorials.mnist import input_data

from deconv import deconv2d
import input as inp
import Batcher

flags = tf.flags
logging = tf.logging

flags.DEFINE_integer("batch_size", 20, "batch size")
flags.DEFINE_integer("updates_per_epoch", 10, "number of updates per epoch")
flags.DEFINE_integer("max_epoch", 1000, "max epoch")
flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
flags.DEFINE_string("working_directory", "", "")
flags.DEFINE_integer("hidden_size", 10, "size of the hidden VAE unit")

tf.app.flags.DEFINE_string('input_path', '../data/tmp/8_pos_delay_3/img/', 'input folder')


FLAGS = flags.FLAGS

pre_hidden_size = 0

def encoder(input_tensor):
  '''Create encoder network.

  Args:
      input_tensor: a batch of flattened images [batch_size, 28*28]

  Returns:
      A tensor that expresses the encoder network
  '''
  x = pt.wrap(input_tensor)
  # print('1', x.tensor.get_shape())
  x = x.conv2d(5, 32, stride=2)
  # print('2', x.tensor.get_shape())
  x = x.conv2d(5, 64, stride=2)
  # print('3', x.tensor.get_shape())
  x = x.conv2d(5, 128, edges='VALID')
  global pre_hidden_size
  pre_hidden_size = int(x.tensor.get_shape()[1])
  # print('4', x.tensor.get_shape())
  x = (x.dropout(0.9).flatten().fully_connected(FLAGS.hidden_size * 2, activation_fn=None)).tensor
  # print('5', x.get_shape())

  return x

  # return (pt.wrap(input_tensor).
  #         conv2d(5, 32, stride=2).
  #         conv2d(5, 64, stride=2).
  #         conv2d(5, 128, edges='VALID').
  #         dropout(0.9).
  #         flatten().
  #         fully_connected(FLAGS.hidden_size * 2, activation_fn=None)).tensor


def decoder(batch_shape, input_tensor=None):
  '''Create decoder network.

      If input tensor is provided then decodes it, otherwise samples from
      a sampled vector.
  Args:
      input_tensor: a batch of vectors to decode

  Returns:
      A tensor that expresses the decoder network
  '''
  epsilon = tf.random_normal([FLAGS.batch_size, FLAGS.hidden_size])
  if input_tensor is None:
    mean = None
    stddev = None
    input_sample = epsilon
  else:
    mean = input_tensor[:, :FLAGS.hidden_size]
    print('mean', mean.get_shape())

    stddev = tf.sqrt(tf.exp(input_tensor[:, FLAGS.hidden_size:]))
    input_sample = mean + epsilon * stddev
  x = pt.wrap(input_sample)
  # print('7', x.tensor.get_shape())
  x = x.reshape([FLAGS.batch_size, 1, 1, FLAGS.hidden_size])
  # print('8', x.tensor.get_shape())
  x = x.deconv2d(pre_hidden_size, 128, edges='VALID')
  # print('9', x.tensor.get_shape())
  x = x.deconv2d(5, 64, edges='VALID')
  # print('10', x.tensor.get_shape())
  x = x.deconv2d(5, 32, stride=2)
  # print('11', x.tensor.get_shape())
  x = x.deconv2d(5, batch_shape[3], stride=2, activation_fn=tf.nn.sigmoid)
  # print('12', x.tensor.get_shape())
  x = x.tensor
  # print('13', x.get_shape())
  return x, mean, stddev
  # return (pt.wrap(input_sample).
  #         reshape([FLAGS.batch_size, 1, 1, FLAGS.hidden_size]).
  #         deconv2d(3, 128, edges='VALID').
  #         deconv2d(5, 64, edges='VALID').
  #         deconv2d(5, 32, stride=2).
  #         deconv2d(5, 1, stride=2, activation_fn=tf.nn.sigmoid)).tensor, mean, stddev


def get_vae_cost(mean, stddev, epsilon=1e-8):
  '''VAE loss
      See the paper

  Args:
      mean:
      stddev:
      epsilon:
  '''
  return tf.reduce_sum(0.5 * (tf.square(mean) + tf.square(stddev) -
                              2.0 * tf.log(stddev + epsilon) - 1.0))


def get_reconstruction_cost(output_tensor, target_tensor, epsilon=1e-8):
  '''Reconstruction loss

  Cross entropy reconstruction loss

  Args:
      output_tensor: tensor produces by decoder
      target_tensor: the target tensor that we want to reconstruct
      epsilon:
  '''
  target_tensor, output_tensor = tf.reshape(target_tensor, [-1]), tf.reshape(output_tensor, [-1])
  return tf.reduce_sum(-target_tensor * tf.log(output_tensor + epsilon) -
                       (1.0 - target_tensor) * tf.log(1.0 - output_tensor + epsilon))


if __name__ == "__main__":
  input_data =  inp.get_images(FLAGS.input_path)
  input_data = input_data[0] / np.max(input_data[0]), input_data[1]
  print(np.max(input_data[0]), np.min(input_data[0]))
  batcher = Batcher.Batcher(FLAGS.batch_size, input_data)
  batch_shape = inp.get_batch_shape(FLAGS.batch_size, FLAGS.input_path)
  loss_n =  (FLAGS.updates_per_epoch * batch_shape[0] * batch_shape[1] *
                       batch_shape[2] * batch_shape[3])
  print(loss_n)
  input_tensor = tf.placeholder(tf.float32, batch_shape)

  with pt.defaults_scope(activation_fn=tf.nn.elu,
                         batch_normalize=True,
                         learned_moments_update_rate=0.0003,
                         variance_epsilon=0.001,
                         scale_after_normalization=True):
    with pt.defaults_scope(phase=pt.Phase.train):
      with tf.variable_scope("model") as scope:
        output_tensor, mean, stddev = decoder(batch_shape, encoder(input_tensor))

    with pt.defaults_scope(phase=pt.Phase.test):
      with tf.variable_scope("model", reuse=True) as scope:
        sampled_tensor, _, _ = decoder(batch_shape)

  vae_loss = get_vae_cost(mean, stddev)
  rec_loss = get_reconstruction_cost(output_tensor, input_tensor)

  loss = vae_loss + rec_loss

  optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0)
  train = pt.apply_optimizer(optimizer, losses=[loss])

  init = tf.initialize_all_variables()

  with tf.Session() as sess:
    sess.run(init)

    for epoch in range(FLAGS.max_epoch):
      print(epoch)
      training_loss = 0.0

      # widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
      # pbar = ProgressBar(maxval=FLAGS.updates_per_epoch, widgets=widgets)
      # pbar.start()
      for i in range(FLAGS.updates_per_epoch):
        x, _ = batcher.get_batch()
        _, loss_value = sess.run([train, loss], {input_tensor: x})
        training_loss += loss_value
        print('%4d/%4d %3d/%3d %f' % (i, FLAGS.updates_per_epoch, epoch, FLAGS.max_epoch,
                                             training_loss))


      training_loss = training_loss / loss_n
      print("Loss %f" % training_loss)

      imgs = sess.run(sampled_tensor)
      for k in range(FLAGS.batch_size):
        imgs_folder = os.path.join(FLAGS.working_directory, 'imgs')
        if not os.path.exists(imgs_folder):
          os.makedirs(imgs_folder)

        imsave(os.path.join(imgs_folder, '%d.png') % k,
               imgs[k])
