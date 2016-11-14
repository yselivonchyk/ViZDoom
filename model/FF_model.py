"""MNIST Autoencoder. """
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import json, os, re, math
import numpy as np
import utils as ut
import input as inp
import tools.checkpoint_utils as ch_utils
import activation_functions as act
import visualization as vis
import prettytensor as pt
import Model

tf.app.flags.DEFINE_integer('stride', 1, 'Data is permuted in series of INT consecutive inputs')
FLAGS = tf.app.flags.FLAGS

DEV = False


def _get_stats_template():
  return {
    'batch': [],
    'input': [],
    'encoding': [],
    'reconstruction': [],
    'total_loss': 0,
  }


class FF_model(Model.Model):
  model_id = 'do'
  decoder_scope = 'dec'
  encoder_scope = 'enc'

  layers = [40, 6, 40]
  layer_narrow = 1

  _image_shape = None
  _batch_shape = None

  # placeholders
  _input = None
  _encoding = None

  _reconstruction = None


  # operations
  _encode = None
  _decode = None
  _reco_loss = None
  _optimizer = None
  _train = None

  _step = None
  _current_step = None
  _visualize_op = None

  def __init__(self,
               weight_init=None,
               activation=act.sigmoid,
               optimizer=tf.train.AdamOptimizer):
    self._weight_init = weight_init
    self._activation = activation
    self._optimizer_constructor = optimizer
    if FLAGS.load_from_checkpoint:
      self.load_meta(FLAGS.load_from_checkpoint)

  def get_layer_info(self):
    return self.layers

  def get_meta(self, meta=None):
    meta = super(FF_model, self).get_meta(meta=meta)
    # meta['seq'] = FLAGS.stride
    return meta

  def load_meta(self, save_path):
    meta = super(FF_model, self).load_meta(save_path)
    self._weight_init = meta['init']
    self._optimizer = tf.train.AdadeltaOptimizer \
      if 'Adam' in meta['opt'] \
      else tf.train.AdadeltaOptimizer
    self._activation = act.sigmoid
    self.set_layer_sizes(meta['h'])
    # FLAGS.stride = int(meta['str']) if 'str' in meta else 2
    ut.configure_folders(FLAGS, self.get_meta())
    return meta

  # MODEL

  def build_model(self):
    with tf.device('/cpu:0'):
      tf.reset_default_graph()
      self._batch_shape = inp.get_batch_shape(FLAGS.batch_size, FLAGS.input_path)
      self._current_step = tf.Variable(0, trainable=False, name='global_step')
      self._step = tf.assign(self._current_step, self._current_step + 1)
      with pt.defaults_scope(activation_fn=self._activation.func):
        with pt.defaults_scope(phase=pt.Phase.train):
          with tf.variable_scope(self.encoder_scope):
            self._build_encoder()
          with tf.variable_scope(self.decoder_scope):
            self._build_decoder()

  def _build_encoder(self):
    """Construct encoder network: placeholders, operations, optimizer"""
    self._input = tf.placeholder(tf.float32, self._batch_shape, name='input')

    self._encode = (pt.wrap(self._input)
                    .flatten())

    for i in range(self.layer_narrow + 1):
      size, desc = self.layers[i], 'enc_hidden_%d' % i
      self._encode = self._encode.fully_connected(size, name=desc)
      if i == self.layer_narrow-1:
        print('dropout:', i, self.layers[i], 1.0-FLAGS.dropout)
        self._encode = self._encode.dropout(1.0-FLAGS.dropout)

  def _build_decoder(self, weight_init=tf.truncated_normal):
    narrow, layers = self.layers[self.layer_narrow], self.layers[self.layer_narrow+1:]

    self._encoding = tf.placeholder(tf.float32, (FLAGS.batch_size, narrow), name='encoding')
    self._reconstruction = tf.placeholder(tf.float32, self._batch_shape)

    for i, size in enumerate(layers):
      start = self._decode if i != 0 else self._encode
      self._decode = start.fully_connected(size, name='enc_hidden_%d' % i)

    self._decode = (self._decode
        .fully_connected(np.prod(self._image_shape), init=weight_init, name='output')
        .reshape(self._batch_shape))

    self._reco_loss = self._build_reco_loss(self._reconstruction)
    self._optimizer = self._optimizer_constructor(learning_rate=FLAGS.learning_rate)
    self._train = self._optimizer.minimize(self._reco_loss)

  # DATA
  @ut.timeit
  def fetch_datasets(self, activation_func_bounds):
    original_data, filters = inp.get_images(FLAGS.input_path)
    ut.print_info('shapes. data, filters: %s' % str((original_data.shape, filters.shape)))

    original_data = inp.rescale_ds(original_data, activation_func_bounds.min, activation_func_bounds.max)
    self._image_shape = inp.get_image_shape(FLAGS.input_path)

    if DEV:
      original_data, filters = original_data[:300], filters[:300]

    self.epoch_size = math.ceil(len(original_data) / FLAGS.batch_size)
    self.test_size = math.ceil(len(original_data) / FLAGS.batch_size)
    return original_data, filters

  # @ut.timeit
  def _get_epoch_dataset(self):
    ds, filters = self._get_blurred_dataset(), self._filters
    ds = inp.pad_set(ds, FLAGS.batch_size)
    filters = inp.pad_set(ds, FLAGS.batch_size)
    # permute
    (train_set, filters), permutation = inp.permute_data_in_series((ds, filters), FLAGS.stride)
    # (train_set, filters), permutation = (ds, filters), np.arange(len(ds))
    # construct feed
    feed = pt.train.feed_numpy(FLAGS.batch_size, train_set, filters)
    return feed, permutation

  def set_layer_sizes(self, h):
    if isinstance(h, str):
      ut.print_info('new layer sizes: %s' % h)
      h = h.replace('/', '|')
      h = list(map(int, h.split('|')))
    self.layers = h
    self.layer_narrow = np.argmin(h)
    print(self.layers, self.layer_narrow)

  # TRAIN

  def train(self, epochs_to_train=5):
    meta = self.get_meta()
    ut.print_time('train started: \n%s' % ut.to_file_name(meta))
    # return meta, np.random.randn(epochs_to_train)
    ut.configure_folders(FLAGS, meta)

    self._dataset, _ = self.fetch_datasets(self._activation)
    self._filters = np.zeros(len(self._dataset))
    self.build_model()
    self._register_training_start()

    with tf.Session() as sess:
      sess.run(tf.initialize_all_variables())
      self._saver = tf.train.Saver()

      if FLAGS.load_state and os.path.exists(self.get_checkpoint_path()):
        self._saver.restore(sess, self.get_checkpoint_path())
        ut.print_info('Restored requested. Previous epoch: %d' % self.get_past_epochs(), color=31)

      # MAIN LOOP
      for current_epoch in xrange(epochs_to_train):
        feed, permutation = self._get_epoch_dataset()
        for _, batch in enumerate(feed):

          encoding, reconstruction, loss, _, _ = sess.run(
            [self._encode, self._decode, self._reco_loss, self._train, self._step],
            feed_dict={self._input: batch[0], self._reconstruction: batch[0]})

          self._register_batch(batch, encoding, reconstruction, loss)
        self._register_epoch(current_epoch, epochs_to_train, permutation, sess)
      self._writer = tf.train.SummaryWriter(FLAGS.logdir, sess.graph)
      meta = self._register_training()
    return meta, self._stats['epoch_accuracy']


if __name__ == '__main__':
  # FLAGS.load_from_checkpoint = './tmp/doom_bs__act|sigmoid__bs|20__h|500|5|500__init|na__inp|cbd4__lr|0.0004__opt|AO'
  FLAGS.input_path = '../data/tmp_grey/romb8.2.2/img/'

  epochs = 100
  import sys

  model = FF_model()
  args = dict([arg.split('=', maxsplit=1) for arg in sys.argv[1:]])
  print(args)
  if len(args) == 1:
    ut.print_info('DEV mode', color=33)
    args['input'] = 'tmp/romb8.2.2/img'
    FLAGS.blur_sigma = 0
    ut.print_info(args['input'],  color=33)

  if 'epochs' in args:
    epochs = int(args['epochs'])
    ut.print_info('epochs: %d' % epochs, color=36)
  if 'stride' in args:
    FLAGS.stride = int(args['stride'])
  if 'blur' in args:
    FLAGS.blur_sigma = int(args['blur'])
  if 'suffix' in args:
    FLAGS.suffix = args['suffix']
  if 'input' in args:
    if args['input'][0] != '/':
      args['input'] = '/' + args['input']
    if 'tmp' not in args['input']:
      args['input'] = '/tmp' + args['input']

    parts = FLAGS.input_path.split('/')[:-4]
    FLAGS.input_path = '/'.join(parts) + args['input']
    ut.print_info('input: %s' % FLAGS.input_path, color=36)
  if 'h' in args:
    model.set_layer_sizes(args['h'])

  all_data = [x[0] for x in os.walk( '../data/tmp_grey/') if 'img' in x[0]]
  # for _, path in enumerate(all_data):
  #   print(path)
  #   FLAGS.input_path = path
  model.train(epochs)