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

tf.app.flags.DEFINE_string('input_path', '../data/tmp/360/img/', 'input folder')
tf.app.flags.DEFINE_integer('batch_size', 30, 'Batch size')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Create visualization of ')
tf.app.flags.DEFINE_integer('stride', 2, 'Data is permuted in series of INT consecutive inputs')

tf.app.flags.DEFINE_string('suffix', 'run', 'Suffix to use to distinguish models by purpose')

tf.app.flags.DEFINE_string('save_path', './tmp/checkpoint', 'Where to save the model checkpoints.')
tf.app.flags.DEFINE_string('logdir', '', 'where to save logs.')

tf.app.flags.DEFINE_boolean('visualize', True, 'Create visualization of decoded images')
tf.app.flags.DEFINE_integer('vis_substeps', 10, 'Use INT intermediate images')

tf.app.flags.DEFINE_boolean('load_state', True, 'Create visualization of ')
tf.app.flags.DEFINE_integer('save_every', 1000, 'Save model state every INT epochs')
tf.app.flags.DEFINE_integer('acc_every', 2, 'Save model state every INT epochs')
tf.app.flags.DEFINE_integer('save_encodings_every', 250, 'Save model state every INT epochs')

tf.app.flags.DEFINE_string('load_from_checkpoint', None, 'where to save logs.')

FLAGS = tf.app.flags.FLAGS


class DoomModel:
  model_id = 'bs'
  _epoch_size = None
  _test_size = None

  layer_narrow = 2
  layer_encoder = 1000
  layer_decoder = 1000

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
    self._weight_init = weight_init
    self._activation = activation
    self._optimizer = optimizer
    if FLAGS.load_from_checkpoint:
      self.load_meta(FLAGS.load_from_checkpoint)

  def encoder(self, input_tensor):
    return (pt.wrap(input_tensor)
            .flatten()
            .fully_connected(self.layer_encoder)
            .fully_connected(self.layer_narrow))

  # def encoder_conv(self, input_tensor):
  #   return (pt.wrap(input_tensor)
  #           .flatten()
  #           .fully_connected(self.layer_encoder)
  #           .fully_connected(self.layer_narrow))

  def decoder(self, input_tensor=None, weight_init=tf.truncated_normal):
    return (pt.wrap(input_tensor)
            .fully_connected(self.layer_decoder)
            .fully_connected(self._image_shape[0] * self._image_shape[1] * self._image_shape[2],
                             init=weight_init))

  # MISC

  def get_layer_info(self):
    return [self.layer_encoder, self.layer_narrow, self.layer_decoder]

  def process_in_batches(self, session, placeholders, op, set, batch_size=None):
    batch_size = batch_size if batch_size else FLAGS.batch_size
    batch_count = int(len(set) / batch_size)
    # if batch_count*batch_size != len(set):
    #   ut.print_info('not all examples are going to be processed: %d/%d' % (
    #     batch_count*batch_size, len(set)))
    feed = zip(xrange(batch_count), pt.train.feed_numpy(batch_size, set, set))
    result_batches = [session.run([op], dict(zip(placeholders, data)))[0] for _, data in feed]
    return np.vstack(result_batches)

  # META

  def get_meta(self, meta=None):
    meta = meta if meta else {}

    meta['postf'] = self.model_id
    meta['a'] = 's'
    meta['lr'] = FLAGS.learning_rate
    meta['init'] = self._weight_init
    meta['bs'] = FLAGS.batch_size
    meta['h'] = self.get_layer_info()
    meta['opt'] = self._optimizer
    meta['seq'] = FLAGS.stride
    meta['inp'] = inp.get_input_name(FLAGS.input_path)
    return meta

  def save_meta(self):
    meta = self.get_meta()
    ut.configure_folders(FLAGS, meta)
    meta['a'] = 's'
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
    FLAGS.stride = int(meta['str']) if 'str' in meta else 2
    self._weight_init = meta['init']
    self._optimizer = tf.train.AdadeltaOptimizer \
      if 'Adam' in meta['opt'] \
      else tf.train.AdadeltaOptimizer
    self._activation = act.sigmoid
    self.layer_encoder = meta['h'][0]
    self.layer_narrow = meta['h'][1]
    self.layer_decoder = meta['h'][2]
    FLAGS.load_state = True
    ut.configure_folders(FLAGS, self.get_meta())

  # MODEL

  def build_model(self):
    tf.reset_default_graph()
    batch_shape = inp.get_batch_shape(FLAGS.batch_size, FLAGS.input_path)
    self._input_placeholder = tf.placeholder(tf.float32, batch_shape)
    self._output_placeholder = tf.placeholder(tf.float32, batch_shape)
    self._encoding_placeholder = tf.placeholder(tf.float32, (FLAGS.batch_size, self.layer_narrow))

    with pt.defaults_scope(activation_fn=self._activation.func):
      with pt.defaults_scope(phase=pt.Phase.train):
        with tf.variable_scope("model"):
          self._encode_op = self.encoder(self._input_placeholder)
          self._encdec_op = self.decoder(
            self._encode_op,
            weight_init=self._weight_init)
          self._visualize_op = self._encdec_op.reshape(batch_shape)\
            .apply(tf.mul, 255)\
            .apply(tf.cast, tf.uint8)

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
          .reshape((1, image_shape[0], image_shape[1], image_shape[2])))
      self._decode_op = raw_decoding_op.apply(tf.mul, 255).apply(tf.cast, tf.uint8)
      # self._decode_op = tf.cast(tf.mul(raw_decoding_op, tf.constant(255.)), tf.uint8)

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
    var = ch_utils.load_variable( tf.train.latest_checkpoint(FLAGS.load_from_checkpoint), name)
    return var

  # TRAIN

  def get_reconstruction_cost(self, output_tensor, target_tensor, epsilon=1e-8):
    target_tensor = (pt.wrap(target_tensor).flatten()).tensor
    return -tf.reduce_sum(output_tensor * tf.log(target_tensor))

  def loss(self, original, reconstruction):
    pretty_input = pt.wrap(original)
    reconstruction = pt.wrap(reconstruction).flatten()
    pt.wrap(tf.reduce_mean(tf.square(original - reconstruction)))
    return pretty_input.cross_entropy(reconstruction)

  def square_loss(self, output_tensor, output_actual):
    return output_tensor.l2_regression(pt.wrap(output_actual).flatten())

  def fetch_datasets(self, activation_func_bounds):
    original_data, labels = inp.get_images(FLAGS.input_path)
    original_data = inp.rescale_ds(original_data, activation_func_bounds.min, activation_func_bounds.max)
    # original_data, labels = original_data[:120], labels[:120]
    # print(original_data.shape, labels.shape)
    global _epoch_size, _test_size
    self._image_shape = inp.get_image_shape(FLAGS.input_path)
    _epoch_size = len(original_data) // FLAGS.batch_size
    _test_size = len(original_data) // FLAGS.batch_size

    visual_set, _ = data_utils.permute_data((original_data, labels))
    return original_data, visual_set[0:FLAGS.batch_size]


  def set_layer_sizes(self, h):
    self.layer_encoder = h[0]
    self.layer_narrow = h[1]
    self.layer_decoder = h[2]

  @staticmethod
  def get_past_epochs():
    return int(bookkeeper.global_step().eval() / _epoch_size)

  def save_encodings(self, encodings, visual_set, reconstruction, accuracy):
    epochs_past = DoomModel.get_past_epochs()
    meta = {'suf': 'encodings', 'e': int(epochs_past), 'er': int(accuracy)}
    projection_file = ut.to_file_name(meta, FLAGS.save_path, 'txt')
    np.savetxt(projection_file, encodings)
    vis.visualize_encoding(encodings, FLAGS.save_path, meta, visual_set, reconstruction)

  def checkpoint(self, runner, sess):
    self.save_meta()
    runner._saver.max_to_keep = 2
    runner._saver.save(sess, FLAGS.save_path, 9999)

  def print_epoch_info(self, accuracy, current_epoch, reconstructions, epochs):
    epochs_past = DoomModel.get_past_epochs() - current_epoch
    reconstruction_info = ''
    accuracy_info = '' if accuracy is None else 'accuracy %0.5f' % accuracy
    if FLAGS.visualize and DoomModel.is_stopping_point(current_epoch, epochs,
                                          stop_every=FLAGS.vis_substeps):
      reconstruction_info = 'last reconstruction: (min, max): (%3d %3d)' % (
      np.min(reconstructions[-1]),
      np.max(reconstructions[-1]))
    epoch_past_info = '' if epochs_past is None else '+%d' % (epochs_past - 1)
    info_string = 'Epochs %2d/%d%s %s %s' % (
      current_epoch + 1,
      epochs,
      epoch_past_info,
      accuracy_info,
      reconstruction_info)
    ut.print_time(info_string)

  @staticmethod
  def is_stopping_point(current_epoch, epochs_to_train, stop_every=None, stop_x_times=None,
                      stop_on_last=True):
    if stop_on_last and current_epoch + 1 == epochs_to_train:
      return True
    if stop_x_times is not None:
      return current_epoch % np.ceil(epochs_to_train / float(FLAGS.vis_substeps)) == 0
    if stop_every is not None:
      return (current_epoch+1) % stop_every == 0

  def train(self, epochs_to_train=5):
    meta = self.get_meta()
    ut.print_time('train started: \n%s' % ut.to_file_name(meta))
    # return meta, np.random.randn(epochs_to_train)
    ut.configure_folders(FLAGS, meta)
    accuracy_by_epoch, epoch_reconstruction = [], []

    original_set, visual_set = self.fetch_datasets(self._activation)
    padding_length = FLAGS.batch_size - (len(original_set) % FLAGS.batch_size)
    encoding_set = np.concatenate((original_set, original_set[:padding_length]))
    self.build_model()
    placeholders = (self._input_placeholder, self._output_placeholder)
    _runner = pt.train.Runner(save_path=FLAGS.save_path, logdir=FLAGS.logdir)

    with tf.Session() as sess:
      sess.run(tf.initialize_all_variables())

      if FLAGS.load_state:
        _runner.load_from_checkpoint(sess)
        ut.print_info('Restored requested. Previous epoch: %d' % self.get_past_epochs(), color=31)

      for current_epoch in xrange(epochs_to_train):
        train_set = inp.permute_array_in_series(original_set, FLAGS.stride)

        if FLAGS.visualize and DoomModel.is_stopping_point(
          current_epoch, epochs_to_train, stop_x_times=FLAGS.vis_substeps):
          epoch_reconstruction.append(self.process_in_batches(
            sess, (self._input_placeholder, self._output_placeholder), self._visualize_op, visual_set))

        _runner.train_model(
          self._train_op,
          self._loss,
          _epoch_size,
          feed_vars=placeholders,
          feed_data=pt.train.feed_numpy(FLAGS.batch_size, train_set, train_set),
          print_every=None)

        accuracy = None
        if DoomModel.is_stopping_point(current_epoch, epochs_to_train, FLAGS.acc_every):
          decodings = self.process_in_batches(
            sess,
            (self._input_placeholder, self._output_placeholder),
            self._encdec_op, original_set)
          accuracy = np.sqrt(np.square(decodings - original_set[:len(decodings)].reshape(decodings.shape)).mean())
          accuracy_by_epoch.append(accuracy)

        if DoomModel.is_stopping_point(current_epoch, epochs_to_train, FLAGS.save_every):
          self.checkpoint(_runner, sess)
        if DoomModel.is_stopping_point(current_epoch, epochs_to_train, FLAGS.save_encodings_every):
          encoding = self.process_in_batches(sess, placeholders, self._encode_op, encoding_set)
          encoding = encoding[:len(original_set)]
          visual_reconstruction = self.process_in_batches(
            sess, (self._input_placeholder, self._output_placeholder), self._visualize_op, visual_set)
          self.save_encodings(encoding, visual_set, visual_reconstruction, accuracy)

        self.print_epoch_info(accuracy, current_epoch, epoch_reconstruction, epochs_to_train)

      meta['acu'] = int(np.min(accuracy_by_epoch))
      meta['e'] = self.get_past_epochs()
      ut.reconstruct_images_epochs(np.asarray(epoch_reconstruction), visual_set,
                                   save_params=meta, img_shape=self._image_shape)

    ut.print_time('Best Quality: %f for %s' % (np.min(accuracy_by_epoch), ut.to_file_name(meta)))
    return meta, accuracy_by_epoch


def parse_params():
  params = {}
  for i, param in enumerate(sys.argv):
    if '-' in param:
      params[param[1:]] = sys.argv[i+1]
  print(params)
  return params


if __name__ == '__main__':
  # FLAGS.load_from_checkpoint = './tmp/doom_bs__act|sigmoid__bs|20__h|500|5|500__init|na__inp|cbd4__lr|0.0004__opt|AO'
  epochs = 100
  import sys

  model = DoomModel()
  args = dict([arg.split('=', maxsplit=1) for arg in sys.argv[1:]])
  print(args)
  if 'epochs' in args:
    epochs = int(args['epochs'])
    ut.print_info('epochs: %d' % epochs, color=36)
  if 'stride' in args:
    FLAGS.stride = int(args['stride'])
  if 'input' in args:
    parts = FLAGS.input_path.split('/')
    parts[-3] = args['input']
    FLAGS.input_path = '/'.join(parts)
    ut.print_info('input %s' % FLAGS.input_path, color=36)
  if 'h' in args:
    layers = list(map(int, args['h'].split('/')))
    ut.print_info('layers %s' % str(layers), color=36)
    model.set_layer_sizes(layers)

  model.train(epochs)


  # epochs = 5
  # # FLAGS.input_path = '../data/tmp/8_pos_delay/img/'
  # # model = DoomModel()
  # model.set_layer_sizes([500, 2, 500])
  # model.train(epochs)

  # epochs = 5000
  # FLAGS.input_path = '../data/tmp/8_pos_delay_3/img/'
  # model = DoomModel()
  # model.set_layer_sizes([500, 3, 500])
  # model.train(epochs)
  #
  # epochs = 5000
  # FLAGS.input_path = '../data/tmp/8_pos_delay/img/'
  # model = DoomModel()
  # model.set_layer_sizes([500, 8, 500])
  # model.train(epochs)