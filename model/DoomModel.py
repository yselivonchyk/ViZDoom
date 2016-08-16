"""MNIST Autoencoder. """

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
import prettytensor.bookkeeper as bookkeeper
from prettytensor.tutorial import data_utils

tf.app.flags.DEFINE_string('save_path', './tmp/checkpoint', 'Where to save the model checkpoints.')
tf.app.flags.DEFINE_string('logdir', '', 'where to save logs.')

tf.app.flags.DEFINE_boolean('visualize', True, 'Create visualization of ')
tf.app.flags.DEFINE_integer('vis_substeps', 10, 'Use INT intermediate images')

tf.app.flags.DEFINE_boolean('load_state', True, 'Create visualization of ')
tf.app.flags.DEFINE_integer('save_every', 50, 'Save model state every INT epochs')
tf.app.flags.DEFINE_integer('batch_size', 30, 'Batch size')

tf.app.flags.DEFINE_float('learning_rate', 0.0004, 'Create visualization of ')
tf.app.flags.DEFINE_string('input_path', '../data/tmp/circle_basic_delay_4/img/', 'path to the '
                                                                                'source folder')
tf.app.flags.DEFINE_integer('series_length', 3, 'Data is permuted in series of INT consecutive inputs')

tf.app.flags.DEFINE_string('load_from_checkpoint', None, 'where to save logs.')


FLAGS = tf.app.flags.FLAGS


class DoomModel:
  _epoch_size = None
  _test_size = None

  layer_narrow = 5
  layer_encoder = 500
  layer_decoder = 500

  _image_shape = None
  _batch_shape = None

  _input_placeholder = None
  _output_placeholder = None
  _encoding_placeholder = None

  _encdec_op = None
  _encode_op = None
  _decode_op = None
  _train_op = None
  _loss = None
  _visualize_op = None

  def __init__(self,
              weight_init=None,
              activation=act.sigmoid,
              optimizer=tf.train.AdamOptimizer):

    self._encoder_to_use = self.encoder_2
    self._decoder_to_use = self.decoder_2
    self._weight_init = weight_init
    self._activation= activation
    self._optimizer= optimizer
    if FLAGS.load_from_checkpoint:
      self.load_meta(FLAGS.load_from_checkpoint)

  def encoder(self, input_tensor):
    return (pt.wrap(input_tensor)
            .flatten()
            .fully_connected(self.layer_narrow)
            ).tensor

  def encoder_2(self, input_tensor):
    return (pt.wrap(input_tensor)
            .flatten()
            .fully_connected(self.layer_encoder)
            .fully_connected(self.layer_narrow))

  def encoder_conv(self, input_tensor):
    return (pt.wrap(input_tensor)
            .flatten()
            .fully_connected(self.layer_encoder)
            .fully_connected(self.layer_narrow))

  def decoder(self, input_tensor=None, weight_init=tf.truncated_normal):
    return (pt.wrap(input_tensor)
      .fully_connected(
      _image_shape[0] * _image_shape[1] * _image_shape[2],
      init=weight_init))

  def decoder_2(self, input_tensor=None, weight_init=tf.truncated_normal):
    return (pt.wrap(input_tensor)
            .fully_connected(self.layer_decoder)
            .fully_connected(_image_shape[0] * _image_shape[1] * _image_shape[2],
                             init=weight_init))

  # MISC

  def get_layer_info(self):
    return [self.layer_encoder, self.layer_narrow, self.layer_decoder]

  def process_in_batches(self, session, placeholders, op, set, batch_size=None):
    batch_size = batch_size if batch_size else FLAGS.batch_size
    batch_count = int(len(set) / batch_size)
    if batch_count*batch_size != len(set):
      ut.print_info('not all examples are going to be processed: %d/%d' % (
        batch_count*batch_size, len(set)))
    feed = zip(xrange(batch_count), pt.train.feed_numpy(batch_size, set, set))
    result_batches = [session.run([op], dict(zip(placeholders, data)))[0] for _, data in feed]
    return np.vstack(result_batches)

  # META

  def get_meta(self, meta=None):
    meta = meta if meta else {}

    meta['suf'] = 'doom_bs'
    meta['act'] = self._activation.func
    meta['lr'] = FLAGS.learning_rate
    meta['init'] = self._weight_init
    meta['bs'] = FLAGS.batch_size
    meta['h'] = self.get_layer_info()
    meta['opt'] = self._optimizer
    meta['inp'] = inp.get_input_name(FLAGS.input_path)
    return meta

  def save_meta(self):
    meta = self.get_meta()
    ut.configure_folders(FLAGS, meta)
    meta['act'] = str(meta['act']).split(' ')[1]
    meta['opt'] = str(meta['opt']).split('.')[-1][:-2]
    meta['input_path'] = FLAGS.input_path
    path = os.path.join(FLAGS.save_path, 'meta.txt')
    json.dump(meta, open(path,'w'))

  def load_meta(self, save_path):
    path = os.path.join(save_path, 'meta.txt')
    meta = json.load(open(path, 'r'))
    FLAGS.save_path = save_path
    FLAGS.batch_size = meta['bs']
    FLAGS.input_path = meta['input_path']
    FLAGS.learning_rate = meta['lr']
    self._weight_init = meta['init']
    self._activation = tf.train.AdadeltaOptimizer if 'Adam' in meta['opt'] else tf.train.AdadeltaOptimizer
    self._activation = act.sigmoid if 'sigmoid' in meta['act'] else act.tanh
    self.layer_encoder = meta['h'][0]
    self.layer_narrow = meta['h'][1]
    self.layer_decoder = meta['h'][2]
    FLAGS.load_state = True
    ut.configure_folders(FLAGS, self.get_meta())

  # MODEL

  def build_model(self):
    tf.reset_default_graph()
    self._input_placeholder = tf.placeholder(tf.float32, self.get_batch_shape())
    self._output_placeholder = tf.placeholder(tf.float32, self.get_batch_shape())
    self._encoding_placeholder = tf.placeholder(tf.float32, (FLAGS.batch_size, self.layer_narrow))

    placeholders = (self._input_placeholder, self._output_placeholder)
    with pt.defaults_scope(activation_fn=self._activation.func):
      with pt.defaults_scope(phase=pt.Phase.train):
        with tf.variable_scope("model"):
          self._encode_op = self._encoder_to_use(self._input_placeholder)
          self._encdec_op = self._decoder_to_use(
            self._encode_op,
            weight_init=self._weight_init)
          self._visualize_op = tf.cast(tf.mul(self._encdec_op, tf.constant(255.)), tf.uint8)

    self._loss = self.square_loss(self._encdec_op, self._output_placeholder)
    optimizer = self._optimizer(learning_rate=FLAGS.learning_rate)
    self._train_op = pt.apply_optimizer(optimizer, losses=[self._loss])

  # DECODER

  def build_decoder(self):
    assert FLAGS.load_from_checkpoint
    tf.reset_default_graph()
    w_decoder = self.get_variable('model/fully_connected_2/weights')
    b_decoder = self.get_variable('model/fully_connected_2/bias')
    w_output = self.get_variable('model/fully_connected_3/weights')
    b_output = self.get_variable('model/fully_connected_3/bias')
    image_shape = inp.get_image_shape(FLAGS.input_path)
    self._encoding_placeholder = tf.placeholder(tf.float32, (1, self.layer_narrow))
    with pt.defaults_scope(activation_fn=self._activation.func):
      raw_decoding_op = (
        pt.wrap(self._encoding_placeholder)
          .fully_connected(self.layer_decoder, init=w_decoder, bias_init=b_decoder)
          .fully_connected(image_shape[0] * image_shape[1] * image_shape[2],
                           init=w_output, bias_init=b_output)
          .reshape((1, image_shape[0], image_shape[1], image_shape[2]))).tensor
      self._decode_op = tf.cast(tf.mul(raw_decoding_op, tf.constant(255.)), tf.uint8)

  def decode(self, data):
    assert data.shape[1] == self.layer_narrow

    self.build_decoder()
    with tf.Session() as sess:
      sess.run(tf.initialize_all_variables())

      results = self.process_in_batches(
        sess,
        (self._encoding_placeholder,),
        self._decode_op,
        data,
        batch_size=1)
      return results

  def get_variable(self, name):
    assert FLAGS.load_from_checkpoint
    var = ch_utils.load_variable(FLAGS.save_path, name)
    return var

  # TRAIN

  def get_reconstruction_cost(self, output_tensor, target_tensor, epsilon=1e-8):
    target_tensor = (pt.wrap(target_tensor).flatten()).tensor
    return -tf.reduce_sum(output_tensor * tf.log(target_tensor))

  def loss(self, input, reconstruciton):
    pretty_input = pt.wrap(input)
    reconstruciton = pt.wrap(reconstruciton).flatten()
    pt.wrap(tf.reduce_mean(tf.square(input - reconstruciton)))
    return pretty_input.cross_entropy(reconstruciton)

  def square_loss(self, output_tensor, output_actual):
    return output_tensor.l2_regression(pt.wrap(output_actual).flatten())

  def fetch_datasets(self, activation_func_bounds):
    original_data, labels = inp.get_images(FLAGS.input_path)
    original_data = inp.rescale_ds(original_data, activation_func_bounds.min, activation_func_bounds.max)

    # print(original_data.shape, labels.shape)
    global _epoch_size, _test_size, _image_shape
    _image_shape = inp.get_image_shape(FLAGS.input_path)
    _epoch_size = len(original_data) // FLAGS.batch_size
    _test_size = len(original_data) // FLAGS.batch_size

    visual_set, _ = data_utils.permute_data((original_data, labels))
    return original_data, visual_set[0:FLAGS.batch_size]

  def get_batch_shape(self):
    if len(_image_shape) > 2:
      return FLAGS.batch_size, _image_shape[0], _image_shape[1], _image_shape[2]
    else:
      return FLAGS.batch_size, _image_shape[0], _image_shape[1], [1]

  def checkpoint(self, runner, sess, encoddings, accuracy):
    epochs_past = int(bookkeeper.global_step().eval() / _epoch_size)
    meta = {'suf': 'encodings', 'e': int(epochs_past), 'z_ac': accuracy}
    self.save_meta()
    runner._saver.max_to_keep = 2
    runner._saver.save(sess, FLAGS.save_path, int(epochs_past))
    projection_file = ut.to_file_name(meta, FLAGS.save_path, 'txt')
    np.savetxt(projection_file, encoddings)
    vis.visualize_encoding(encoddings, FLAGS.save_path, meta)

  def get_past_epochs(self):
    return int(bookkeeper.global_step().eval() / _epoch_size)

  def print_epoch_info(self, accuracy, current_epoch, reconstructions, epochs, epochs_past):
    reconstruction_info = ''
    if self.visualization_point(epochs, current_epoch):
      reconstruction_info = 'last reconstruction: (min, max): (%3d %3d)' % (
      np.min(reconstructions[-1]),
      np.max(reconstructions[-1]))
    epoch_past_info = '' if epochs_past is None else '+%d' % epochs_past
    info_string = 'Accuracy after %2d/%d%s epoch(s): %.2f; %s' % (
      current_epoch + 1,
      epochs,
      epoch_past_info,
      accuracy,
      reconstruction_info)
    ut.print_time(info_string)

  def visualization_point(self, epochs_to_train, current_epoch):
    if not FLAGS.visualize:
      return False
    intermidiate_point = current_epoch % np.ceil(epochs_to_train / float(FLAGS.vis_substeps)) == 0
    is_last_step = current_epoch + 1 == epochs_to_train
    return intermidiate_point or is_last_step

  def train(self, epochs_to_train=5):
    meta = self.get_meta()
    ut.configure_folders(FLAGS, meta)
    accuracy_by_epoch, epoch_reconstruction = [], []

    original_set, visual_set = self.fetch_datasets(self._activation)
    self.build_model()
    placeholders = (self._input_placeholder, self._output_placeholder)
    _runner = pt.train.Runner(save_path=FLAGS.save_path, logdir=FLAGS.logdir)

    with tf.Session() as sess:
      sess.run(tf.initialize_all_variables())

      if FLAGS.load_state:
        _runner.load_from_checkpoint(sess)
        epochs_past = self.get_past_epochs()
        ut.print_info('Checkpoint restore requested. previous epoch: %d' % epochs_past, color=31)

      for current_epoch in xrange(epochs_to_train):
        train_set = inp.permute_array_in_series(original_set, FLAGS.series_length)

        if self.visualization_point(epochs_to_train, current_epoch):
          epoch_reconstruction.append(self.process_in_batches(
            sess, (self._input_placeholder, self._output_placeholder), self._visualize_op, visual_set))

        _runner.train_model(
          self._train_op,
          self._loss,
          _epoch_size,
          feed_vars=placeholders,
          feed_data=pt.train.feed_numpy(FLAGS.batch_size, train_set, train_set),
          print_every=None)

        accuracy = _runner.evaluate_model(
          self._loss,
          _test_size,
          feed_vars=placeholders,
          feed_data=pt.train.feed_numpy(FLAGS.batch_size, original_set, original_set))

        self.print_epoch_info(accuracy, current_epoch, epoch_reconstruction,
                              epochs_to_train, epochs_past)
        accuracy_by_epoch.append(accuracy)

        if (current_epoch + 1 == epochs_to_train) or ((current_epoch + 1) % FLAGS.save_every) == 0:
          encoding = self.process_in_batches(sess, placeholders, self._encode_op, original_set)
          self.checkpoint(_runner, sess, encoding, accuracy_by_epoch[-1])

      meta['acu'] = int(np.min(accuracy_by_epoch))
      meta['e'] = self.get_past_epochs()
      ut.reconstruct_images_epochs(np.asarray(epoch_reconstruction), visual_set,
                                   save_params=meta, img_shape=_image_shape)

    ut.print_time('Best Quality: %f for %s' % (np.min(accuracy_by_epoch), ut.to_file_name(meta)))
    return meta, accuracy_by_epoch


if __name__ == '__main__':
  # FLAGS.load_from_checkpoint = './tmp/doom_bs__act|sigmoid__bs|20__h|500|5|500__init|na__inp|cbd4__lr|0.0004__opt|AO'
  model = DoomModel()
  model.train(5)
