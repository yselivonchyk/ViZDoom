"""MNIST Autoencoder. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
import Model as m

tf.app.flags.DEFINE_float('gradient_proportion', 5.0, 'Proportion of gradietn mixture RECO/DRAG')
FLAGS = tf.app.flags.FLAGS

DEV = False


def _clamp(encoding, filter):
  filter_neg = np.ones(len(filter), dtype=filter.dtype) - filter
  # print('\nf_n', filter_neg)
  avg = encoding.mean(axis=0)*filter_neg
  # print('avg', avg, encoding[0])
  # print('avg', encoding.mean(axis=0), encoding[-1], encoding[-1]-avg)
  grad = encoding*filter_neg - avg
  encoding = encoding * filter + avg
  # print('enc', encoding[0], encoding[1])
  # print(np.hstack((encoding, grad)))
  # print('vae', grad[0], grad[1])
  return encoding, grad


def _declamp_grad(vae_grad, reco_grad, filter):
  # print('vae, reco', np.abs(vae_grad).mean(), np.abs((reco_grad*filter)).mean())
  res = vae_grad/FLAGS.gradient_proportion + reco_grad*filter
  # res = vae_grad + reco_grad*filter
  #print('\nvae: %s\nrec: %s\nres %s' % (ut.print_float_list(vae_grad[1]),
  #                                      ut.print_float_list(reco_grad[1]),
  #                                      ut.print_float_list(res[0])))
  return res


class IGN_model(m.Model):
  model_id = 'ignb'
  decoder_scope = 'dec'
  encoder_scope = 'enc'

  layer_narrow = 2
  layer_encoder = 100
  layer_decoder = 100

  _image_shape = None
  _batch_shape = None

  # placeholders
  _input = None
  _encoding = None

  _clamped = None
  _reconstruction = None
  _clamped_grad = None

  # variables
  _clamped_variable = None

  # operations
  _encode = None
  _encoder_loss = None
  _opt_encoder = None
  _train_encoder = None

  _decode = None
  _decoder_loss = None
  _opt_decoder = None
  _train_decoder = None

  _step = None
  _current_step = None
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

  def get_layer_info(self):
    return [self.layer_encoder, self.layer_narrow, self.layer_decoder]

  def get_meta(self, meta=None):
    meta = meta if meta else {}

    meta['postf'] = self.model_id
    meta['a'] = 's'
    meta['lr'] = FLAGS.learning_rate
    meta['init'] = self._weight_init
    meta['bs'] = FLAGS.batch_size
    meta['h'] = self.get_layer_info()
    meta['opt'] = self._optimizer
    meta['inp'] = inp.get_input_name(FLAGS.input_path)
    meta['div'] = '%.1f' % FLAGS.gradient_proportion
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
    FLAGS.gradient_proportion = float(meta['div'])
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
    self._encoding = tf.placeholder(tf.float32, (FLAGS.batch_size, self.layer_narrow), name='encoding')

    self._encode = (pt.wrap(self._input)
                    .flatten()
                    .fully_connected(self.layer_encoder, name='enc_hidden')
                    .fully_connected(self.layer_narrow, name='narrow'))

    # variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.encoder_scope)
    self._encoder_loss = self._encode.l1_regression(pt.wrap(self._encoding))
    ut.print_info('new learning rate: %.8f (%f)' % (FLAGS.learning_rate/FLAGS.batch_size, FLAGS.learning_rate))
    self._opt_encoder = self._optimizer(learning_rate=FLAGS.learning_rate/FLAGS.batch_size)
    self._train_encoder = self._opt_encoder.minimize(self._encoder_loss)

  def _build_decoder(self, weight_init=tf.truncated_normal):
    """Construct decoder network: placeholders, operations, optimizer,
    extract gradient back-prop for encoding layer"""
    self._clamped = tf.placeholder(tf.float32, (FLAGS.batch_size, self.layer_narrow))
    self._reconstruction = tf.placeholder(tf.float32, self._batch_shape)

    clamped_init = np.zeros((FLAGS.batch_size, self.layer_narrow), dtype=np.float32)
    self._clamped_variable = tf.Variable(clamped_init, name='clamped')
    self._assign_clamped = tf.assign(self._clamped_variable, self._clamped)

    # http://stackoverflow.com/questions/40194389/how-to-propagate-gradient-into-a-variable-after-assign-operation
    self._decode = (
      pt.wrap(self._clamped_variable)
        .fully_connected(self.layer_decoder, name='decoder_1')
        .fully_connected(np.prod(self._image_shape), init=weight_init, name='output')
        .reshape(self._batch_shape))

    # variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.decoder_scope)
    self._decoder_loss = self._decode.l2_regression(pt.wrap(self._reconstruction))
    self._opt_decoder = self._optimizer(learning_rate=FLAGS.learning_rate/FLAGS.batch_size)
    self._train_decoder = self._opt_decoder.minimize(self._decoder_loss)

    self._clamped_grad, = tf.gradients(self._decoder_loss, [self._clamped_variable])

  # DATA

  def fetch_datasets(self, activation_func_bounds):
    original_data, filters = inp.get_images(FLAGS.input_path)
    original_data, filters = self.bloody_hack_filterbatches(original_data, filters)
    ut.print_info('shapes. data, filters: %s' % str((original_data.shape, filters.shape)))

    original_data = inp.rescale_ds(original_data, activation_func_bounds.min, activation_func_bounds.max)
    # original_data, labels = original_data[:120], labels[:120]
    # print(original_data.shape, labels.shape)
    self._image_shape = inp.get_image_shape(FLAGS.input_path)

    if DEV:
      original_data = original_data[:300]

    self.epoch_size = math.ceil(len(original_data) / FLAGS.batch_size)
    self.test_size = math.ceil(len(original_data) / FLAGS.batch_size)
    return original_data, filters

  def bloody_hack_filterbatches(self, original_data, filters):
    survivers = np.zeros(len(filters), dtype=np.uint8)
    j, prev = 0, None
    for _, f in enumerate(filters):
      if prev is None or prev[0] == f[0] and prev[1] == f[1]:
        j += 1
      else:
        k = j // FLAGS.batch_size
        for i in range(k):
          start = _ - j + math.ceil(j / k * i)
          survivers[start:start + FLAGS.batch_size] += 1
        # print(j, survivers[_-j:_])
        j = 0
      prev = f
    original_data = np.asarray([x for i, x in enumerate(original_data) if survivers[i] > 0])
    filters = np.asarray([x for i, x in enumerate(filters) if survivers[i] > 0])
    return original_data, filters

  _blurred_dataset, _last_blur_sigma = None, 0

  def _get_blurred_dataset(self):
    epochs_past = self.get_past_epochs()
    if FLAGS.blur_sigma != 0:
      current_sigma = max(0, FLAGS.blur_sigma - int(epochs_past / FLAGS.blur_sigma_decrease))
      if current_sigma != self._last_blur_sigma:
        self._blurred_dataset = inp.apply_gaussian(self._dataset, sigma=current_sigma/10.0)
        self._last_blur_sigma = current_sigma
    return self._blurred_dataset if self._blurred_dataset is not None else self._dataset

  def _get_epoch_dataset(self):
    ds, filters = self._get_blurred_dataset(), self._filters
    # permute
    (train_set, filters), permutation = inp.permute_data_in_series((ds, filters), FLAGS.batch_size, allow_shift=False)
    # construct feed
    feed = pt.train.feed_numpy(FLAGS.batch_size, train_set, filters)
    return feed, permutation

  def set_layer_sizes(self, h):
    self.layer_encoder = h[0]
    self.layer_narrow = h[1]
    self.layer_decoder = h[2]

  def get_past_epochs(self):
    return int(self._current_step.eval() / epoch_size)

  def get_checkpoint_path(self):
    return os.path.join(FLAGS.save_path, '9999.ckpt')

  # TRAIN

  def train(self, epochs_to_train=5):
    meta = self.get_meta()
    ut.print_time('train started: \n%s' % ut.to_file_name(meta))
    # return meta, np.random.randn(epochs_to_train)
    ut.configure_folders(FLAGS, meta)

    self._dataset, self._filters = self.fetch_datasets(self._activation)
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
          filter = batch[1][0]
          assert batch[1][0,0] == batch[1][-1,0]
          encoding, = sess.run([self._encode], feed_dict={self._input: batch[0]})   # 1.1 encode forward
          clamped_enc, vae_grad = _clamp(encoding, filter)                          # 1.2 # clamp

          sess.run(self._assign_clamped, feed_dict={self._clamped:clamped_enc})
          reconstruction, loss, clamped_gradient, _ = sess.run(          # 2.1 decode forward+backward
            [self._decode, self._decoder_loss, self._clamped_grad, self._train_decoder],
            feed_dict={self._clamped: clamped_enc, self._reconstruction: batch[0]})

          declamped_grad = _declamp_grad(vae_grad, clamped_gradient, filter) # 2.2 prepare gradient

          # TODO: invert gradient
          # TODO: sum gradient with differences. Or do just that

          _, step = sess.run(                                            # 3.0 encode backward path
            [self._train_encoder, self._step],
            feed_dict={self._input: batch[0], self._encoding: encoding-declamped_grad})          # Profit
          self._register_batch(batch, encoding, clamped_enc, reconstruction, loss, declamped_grad)
        self._register_epoch(current_epoch, epochs_to_train, permutation, sess)
      self._writer = tf.train.SummaryWriter(FLAGS.logdir, sess.graph)
      meta = self._register_training()
    return meta, self._stats['epoch_accuracy']

  # OUTPUTS

  _epoch_stats = None
  _stats = None

  def _register_training_start(self):
    self._epoch_stats = _get_stats_template()
    self._stats = {
      'epoch_accuracy': [],
      'visual_set': inp.select_random(FLAGS.batch_size, len(self._dataset)),
      'ds_length': len(self._dataset),
      'epoch_reconstructions': [],
      'permutation': None
    }

  def _register_batch(self, batch, encoding, clamped_enc, reconstruction, loss, declamped_drad):
    self._epoch_stats['batch'].append(batch)
    self._epoch_stats['encoding'].append(encoding)
    self._epoch_stats['clamped_enc'].append(clamped_enc)
    self._epoch_stats['reconstruction'].append(reconstruction)
    self._epoch_stats['declamped_drad'].append(declamped_drad)
    self._epoch_stats['total_loss'] += loss

  def _register_epoch(self, epoch, total_epochs, permutation, sess):
    if m.is_stopping_point(epoch, total_epochs, FLAGS.save_every):
      self._saver.save(sess, self.get_checkpoint_path())

    accuracy = 100000 * np.sqrt(self._epoch_stats['total_loss'] / np.prod(self._batch_shape) / epoch_size)
    self._epoch_stats['permutation_reverse'] = np.argsort(permutation)
    visual_set = self._get_visual_set()

    if FLAGS.visualize and m.is_stopping_point(epoch, total_epochs, stop_x_times=FLAGS.vis_substeps):
      self._stats['epoch_reconstructions'].append(visual_set)
    if m.is_stopping_point(epoch, total_epochs, FLAGS.save_encodings_every):
      self.save_encodings(accuracy)
    if m.is_stopping_point(epoch, total_epochs, FLAGS.save_visualization_every):
      self.save_visualization(visual_set, accuracy)
    self._stats['epoch_accuracy'].append(accuracy)

    self.print_epoch_info(accuracy, epoch, self._epoch_stats['reconstruction'][0], total_epochs)
    if epoch + 1 != total_epochs:
      self._epoch_stats = _get_stats_template()

  def _get_visual_set(self):
    r_permutation = self._epoch_stats['permutation_reverse']
    visual_set_indexes = r_permutation[self._stats['visual_set']]
    visual_set = np.vstack(self._epoch_stats['reconstruction'])[visual_set_indexes]
    return visual_set

  @ut.timeit
  def _register_training(self):
    best_acc = np.min(self._stats['epoch_accuracy'])
    meta = self.get_meta()
    meta['acu'] = int(best_acc)
    meta['e'] = self.get_past_epochs()
    original_vis_set = self._dataset[self._stats['visual_set']]
    ut.reconstruct_images_epochs(np.asarray(self._stats['epoch_reconstructions']), original_vis_set,
                                 save_params=meta, img_shape=self._image_shape)
    ut.print_time('Best Quality: %f for %s' % (best_acc, ut.to_file_name(meta)))
    return meta

  def save_encodings(self, accuracy):
    encodings = np.vstack(self._epoch_stats['encoding'][:self._stats['ds_length']])
    encodings = encodings[self._epoch_stats['permutation_reverse']]
    epochs_past = self.get_past_epochs()
    meta = {'suf': 'encodings', 'e': int(epochs_past), 'er': int(accuracy)}
    projection_file = ut.to_file_name(meta, FLAGS.save_path, 'txt')
    np.savetxt(projection_file, encodings)
    return encodings, meta

  def save_visualization(self, reconstruction, accuracy):
    encodings, meta = self.save_encodings(accuracy)
    visual_set = self._get_blurred_dataset()[self._stats['visual_set']]
    vis.visualize_encoding(encodings, FLAGS.save_path, meta, visual_set, reconstruction)

  def print_epoch_info(self, accuracy, current_epoch, reconstruction, epochs):
    epochs_past = self.get_past_epochs() - current_epoch
    reconstruction_info = ''
    accuracy_info = '' if accuracy is None else '| accuracy %d' % int(accuracy)
    if FLAGS.visualize and m.is_stopping_point(current_epoch, epochs,
                                          stop_every=FLAGS.vis_substeps):
      reconstruction_info = '| (min, max): (%3d %3d)' % (
      np.min(reconstruction),
      np.max(reconstruction))
    epoch_past_info = '' if epochs_past is None else '+%d' % (epochs_past - 1)
    info_string = 'Epochs %2d/%d%s %s %s' % (
      current_epoch + 1,
      epochs,
      epoch_past_info,
      accuracy_info,
      reconstruction_info)
    ut.print_time(info_string, same_line=True)


if __name__ == '__main__':
  # FLAGS.load_from_checkpoint = './tmp/doom_bs__act|sigmoid__bs|20__h|500|5|500__init|na__inp|cbd4__lr|0.0004__opt|AO'
  epochs = 300
  import sys

  # x = 100
  # set = np.arange(100)
  # per = np.random.permutation(set)[:10]
  # print(set[per], per, 4)

  model = IGN_model()
  args = dict([arg.split('=', maxsplit=1) for arg in sys.argv[1:]])
  if len(args) == 0:
    global DEV
    DEV = False
    print('DEVELOPMENT MODE ON')
  print(args)
  if 'epochs' in args:
    epochs = int(args['epochs'])
    ut.print_info('epochs: %d' % epochs, color=36)
  if 'sigma' in args:
    FLAGS.sigma = int(args['sigma'])
  if 'suffix' in args:
    FLAGS.suffix = args['suffix']
  if 'input' in args:
    parts = FLAGS.input_path.split('/')
    parts[-3] = args['input']
    FLAGS.input_path = '/'.join(parts)
    ut.print_info('input %s' % FLAGS.input_path, color=36)
  if 'h' in args:
    layers = list(map(int, args['h'].split('/')))
    ut.print_info('layers %s' % str(layers), color=36)
    model.set_layer_sizes(layers)
  if 'divider' in args:
    FLAGS.drag_divider = float(args['divider'])
  if 'lr' in args:
    FLAGS.learning_rate = float(args['lr'])


  model.train(epochs)
