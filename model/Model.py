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


tf.app.flags.DEFINE_string('suffix', 'run', 'Suffix to use to distinguish models by purpose')
tf.app.flags.DEFINE_string('input_path', '../data/tmp_grey/romb8.2.2/img/', 'input folder')
tf.app.flags.DEFINE_string('save_path', './tmp/checkpoint', 'Where to save the model checkpoints.')
tf.app.flags.DEFINE_string('logdir', '', 'where to save logs.')
tf.app.flags.DEFINE_string('load_from_checkpoint', None, 'Load model state from particular checkpoint')

tf.app.flags.DEFINE_integer('save_every', 200, 'Save model state every INT epochs')
tf.app.flags.DEFINE_boolean('load_state', True, 'Load state if possible ')

tf.app.flags.DEFINE_integer('batch_size', 50, 'Batch size')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Create visualization of ')

tf.app.flags.DEFINE_boolean('visualize', True, 'Create visualization of decoded images along training')
tf.app.flags.DEFINE_integer('vis_substeps', 10, 'Use INT intermediate images')

tf.app.flags.DEFINE_integer('save_encodings_every', 20, 'Save model state every INT epochs')
tf.app.flags.DEFINE_integer('save_visualization_every', 40, 'Save model state every INT epochs')

tf.app.flags.DEFINE_integer('blur_sigma', 0, 'Image blur maximum effect')
tf.app.flags.DEFINE_integer('blur_sigma_decrease', 1000, 'Decrease image blur every X epochs')

tf.app.flags.DEFINE_boolean('noise', True, 'apply noise to avoid discretisation')

FLAGS = tf.app.flags.FLAGS

DEV = False


def is_stopping_point(current_epoch, epochs_to_train, stop_every=None, stop_x_times=None,
                      stop_on_last=True):
  if stop_on_last and current_epoch + 1 == epochs_to_train:
    return True
  if stop_x_times is not None:
    return current_epoch % np.ceil(epochs_to_train / float(FLAGS.vis_substeps)) == 0
  if stop_every is not None:
    return (current_epoch + 1) % stop_every == 0


def get_variable(name):
  assert FLAGS.load_from_checkpoint
  var = ch_utils.load_variable(tf.train.latest_checkpoint(FLAGS.load_from_checkpoint), name)
  return var


def get_every_dataset():
  all_data = [x[0] for x in os.walk( '../data/tmp_grey/') if 'img' in x[0]]
  print(all_data)
  return all_data


class Model:
  model_id = 'base'
  epoch_size, test_size = None, None

  _writer, _saver = None, None
  _dataset, _filters = None, None

  def get_layer_info(self):
    return [self.layer_encoder, self.layer_narrow, self.layer_decoder]

  # MODEL

  def build_model(self):
    pass

  def _build_encoder(self):
    pass

  def _build_decoder(self, weight_init=tf.truncated_normal):
    pass

  def _build_reco_loss(self, output_placeholder):
    return self._decode.flatten().l2_regression(pt.wrap(output_placeholder).flatten())

  def train(self, epochs_to_train=5):
    pass

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
    meta['inp'] = inp.get_input_name(FLAGS.input_path)
    return meta

  def save_meta(self, meta=None):
    if meta is None:
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
    FLAGS.load_state = True
    return meta

  # DATA

  _blurred_dataset, _last_blur_sigma = None, 0

  def _get_blurred_dataset(self):
    epochs_past = self.get_past_epochs()
    if FLAGS.blur_sigma != 0:
      current_sigma = max(0, FLAGS.blur_sigma - int(epochs_past / FLAGS.blur_sigma_decrease))
      if current_sigma != self._last_blur_sigma:
        self._blurred_dataset = inp.apply_gaussian(self._dataset, sigma=current_sigma/10.0)
        self._last_blur_sigma = current_sigma
    return self._blurred_dataset if self._blurred_dataset is not None else self._dataset

  # MISC

  def get_past_epochs(self):
    return int(self._current_step.eval() / self.epoch_size)

  def get_checkpoint_path(self):
    return os.path.join(FLAGS.save_path, '9999.ckpt')

  # OUTPUTS

  def _get_stats_template(self):
    return {
      'batch': [],
      'input': [],
      'encoding': [],
      'reconstruction': [],
      'total_loss': 0,
    }

  _epoch_stats = None
  _stats = None

  def _register_training_start(self):
    self._epoch_stats = self._get_stats_template()
    self._stats = {
      'epoch_accuracy': [],
      'visual_set': inp.select_random(FLAGS.batch_size, len(self._dataset)),
      'ds_length': len(self._dataset),
      'epoch_reconstructions': [],
      'permutation': None
    }

  def _register_batch(self, batch, encoding, reconstruction, loss):
    self._epoch_stats['batch'].append(batch)
    self._epoch_stats['encoding'].append(encoding)
    self._epoch_stats['reconstruction'].append(reconstruction)
    self._epoch_stats['total_loss'] += loss

  def _register_epoch(self, epoch, total_epochs, permutation, sess):
    if is_stopping_point(epoch, total_epochs, FLAGS.save_every):
      self._saver.save(sess, self.get_checkpoint_path())

    accuracy = 100000 * np.sqrt(self._epoch_stats['total_loss'] / np.prod(self._batch_shape) / self.epoch_size)
    self._epoch_stats['permutation_reverse'] = np.argsort(permutation)
    visual_set = self._get_visual_set()

    if FLAGS.visualize and is_stopping_point(epoch, total_epochs, stop_x_times=FLAGS.vis_substeps):
      self._stats['epoch_reconstructions'].append(visual_set)
    if is_stopping_point(epoch, total_epochs, FLAGS.save_encodings_every):
      self.save_encodings(accuracy)
    if is_stopping_point(epoch, total_epochs, FLAGS.save_visualization_every):
      self.save_visualization(visual_set, accuracy)
    self._stats['epoch_accuracy'].append(accuracy)

    self.print_epoch_info(accuracy, epoch, self._epoch_stats['reconstruction'][0], total_epochs)
    if epoch + 1 != total_epochs:
      self._epoch_stats = self._get_stats_template()

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
    meta = {'suf': 'encodings', 'e': '%5d' % int(epochs_past), 'er': int(accuracy)}
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
    if FLAGS.visualize and is_stopping_point(current_epoch, epochs,
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