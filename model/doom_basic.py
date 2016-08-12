"""MNIST Autoencoder. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np
import utils as ut
import input as inp
import activation_functions as act
import visualization as vis

import prettytensor as pt
import prettytensor.bookkeeper as bookkeeper
from prettytensor.tutorial import data_utils

BATCH_SIZE = 20
EPOCH_SIZE = 60000 // BATCH_SIZE
TEST_SIZE = 10000 // BATCH_SIZE

LAYER_NARROW = 6
LAYER_ENCODER = 500
LAYER_DECODER = 500

IMAGE_SHAPE = None
BATCH_SHAPE = None

tf.app.flags.DEFINE_string('save_path', './tmp/checkpoint', 'Where to save the model checkpoints.')
tf.app.flags.DEFINE_string('logdir', '', 'where to save logs.')
tf.app.flags.DEFINE_string('model', 'full',
                           'Choose one of the models, either full or conv')
tf.app.flags.DEFINE_string('input_folder',
                           '../data/tmp/circle_basic_delay_4/img/',
                           'Choose input source folder')
tf.app.flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
tf.app.flags.DEFINE_boolean("interactive", True, "use interactive interface")
tf.app.flags.DEFINE_boolean("load_state", True, "load existing model state")
tf.app.flags.DEFINE_integer("save_every", 250, "load existing model state")
FLAGS = tf.app.flags.FLAGS

FLAGS = tf.app.flags.FLAGS


def encoder(input_tensor):
  return (pt.wrap(input_tensor)
          .flatten()
          .fully_connected(LAYER_NARROW)
          ).tensor


def encoder_2(input_tensor):
  return (pt.wrap(input_tensor)
          .flatten()
          .fully_connected(LAYER_ENCODER)
          .fully_connected(LAYER_NARROW))


def encoder_conv(input_tensor):
  return (pt.wrap(input_tensor)
          .flatten()
          .fully_connected(LAYER_ENCODER)
          .fully_connected(LAYER_NARROW))


def decoder(input_tensor=None, weight_init=tf.truncated_normal):
  return (pt.wrap(input_tensor)
    .fully_connected(
    IMAGE_SHAPE[0] * IMAGE_SHAPE[1] * IMAGE_SHAPE[2],
    init=weight_init))


def decoder_2(input_tensor=None, weight_init=tf.truncated_normal):
  return (pt.wrap(input_tensor)
          .fully_connected(LAYER_DECODER)
          .fully_connected(IMAGE_SHAPE[0] * IMAGE_SHAPE[1] * IMAGE_SHAPE[2],
                           init=weight_init))


def get_reconstruction_cost(output_tensor, target_tensor, epsilon=1e-8):
  target_tensor = (pt.wrap(target_tensor).flatten()).tensor
  return -tf.reduce_sum(output_tensor * tf.log(target_tensor))


def loss(input, reconstruciton):
  pretty_input = pt.wrap(input)
  reconstruciton = pt.wrap(reconstruciton).flatten()
  pt.wrap(tf.reduce_mean(tf.square(input - reconstruciton)))
  return pretty_input.cross_entropy(reconstruciton)


def square_loss(output_tensor, output_actual):
  return output_tensor.l2_regression(pt.wrap(output_actual).flatten())


def assert_model(input_placeholder, output_placeholder, test_data, train_data, visualization_data):
  assert visualization_data.shape == input_placeholder.get_shape()
  assert len(train_data.shape) == len(input_placeholder.get_shape())
  assert len(test_data.shape) == len(input_placeholder.get_shape())
  assert visualization_data.shape == output_placeholder.get_shape()
  assert len(train_data.shape) == len(output_placeholder.get_shape())
  assert len(test_data.shape) == len(output_placeholder.get_shape())


def fetch_datasets(data_min, data_max, source_folder):
  global EPOCH_SIZE, TEST_SIZE
  original_data = np.asarray(inp.get_images(source_folder))
  input_source, _ = data_utils.permute_data((original_data, np.zeros(len(original_data))))

  EPOCH_SIZE = len(input_source) // BATCH_SIZE
  TEST_SIZE = len(input_source) // BATCH_SIZE

  input_source = ut.rescale_ds(input_source, data_min, data_max)
  return input_source, original_data, input_source[0:BATCH_SIZE]


def get_batch_shape():
  if len(IMAGE_SHAPE) > 2:
    return BATCH_SIZE, IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2]
  else:
    return BATCH_SIZE, IMAGE_SHAPE[0], IMAGE_SHAPE[1], [1]


def checkpoint(runner, sess, encoddings, accuracy):
  epochs_past = int(bookkeeper.global_step().eval() / EPOCH_SIZE)
  meta = {'suf': 'encodings', 'e': int(epochs_past), 'z_ac': accuracy}
  runner._saver.max_to_keep = 2
  runner._saver.save(sess, FLAGS.save_path, int(epochs_past))
  projection_file = ut.to_file_name(meta, FLAGS.save_path, 'txt')
  np.savetxt(projection_file, encoddings)
  vis.visualize_encoding(encoddings, FLAGS.save_path, meta)


def get_layer_info():
  return [LAYER_ENCODER, LAYER_NARROW, LAYER_DECODER]


visualization_substeps = 10


def main(_=None,
         weight_init=None,
         activation=act.sigmoid,
         epochs_to_train=10,
         learning_rate=0.0004,
         source_folder=FLAGS.input_folder,
         enc=encoder_2,
         dec=decoder_2,
         optimizer=tf.train.AdamOptimizer):
  tf.reset_default_graph()

  global IMAGE_SHAPE
  accuracy_by_epoch = []
  epoch_reconstruction = []
  IMAGE_SHAPE = inp.get_image_shape(source_folder)
  input_name = FLAGS.input_folder.split('/')[-3]
  meta = {'suf': 'doom_bs', 'act': activation.func, 'lr': learning_rate,
          'init': weight_init, 'bs': BATCH_SIZE, 'h': get_layer_info(), 'opt': optimizer,
          'inp': input_name}
  ut.configure_folders(FLAGS, meta)

  input_placeholder = tf.placeholder(tf.float32, get_batch_shape())
  output_placeholder = tf.placeholder(tf.float32, get_batch_shape())

  train_data, test_data, visualization_data = \
    fetch_datasets(activation.min, activation.max, source_folder)
  assert_model(input_placeholder, output_placeholder, test_data, train_data, visualization_data)

  with pt.defaults_scope(activation_fn=activation.func,
                         # batch_normalize=True,
                         # learned_moments_update_rate=0.0003,
                         # variance_epsilon=0.001,
                         # scale_after_normalization=True
                         ):
    with pt.defaults_scope(phase=pt.Phase.train):
      with tf.variable_scope("model"):
        encode_op = enc(input_placeholder)
        output_tensor = dec(encode_op, weight_init=weight_init)

  pretty_loss = square_loss(output_tensor, output_placeholder)
  # pretty_loss = square_loss(output_tensor, output_placeholder)
  optimizer = optimizer(learning_rate=learning_rate)
  train = pt.apply_optimizer(optimizer, losses=[pretty_loss])
  init = tf.initialize_all_variables()
  runner = pt.train.Runner(save_path=FLAGS.save_path, logdir=FLAGS.logdir)

  with tf.Session() as sess:
    sess.run(init)

    if FLAGS.load_state:
      runner.load_from_checkpoint(sess)
      epochs_past = int(bookkeeper.global_step().eval() / EPOCH_SIZE)
      ut.print_info('Checkpoint restore requested. previous epoch: %d' % epochs_past, color=31)

    for train_epoch in xrange(epochs_to_train):
      epoch_info = ''

      if train_epoch % np.ceil(epochs_to_train / float(visualization_substeps)) == 0 or train_epoch + 1 == epochs_to_train:
        reconstruct, loss_value = sess.run([output_tensor, pretty_loss],
                                           {input_placeholder: visualization_data,
                                            output_placeholder: visualization_data})
        epoch_reconstruction.append(reconstruct)
        epoch_info += 'epoch:%d (min, max): (%f %f)' % \
                      (train_epoch, np.min(reconstruct), np.max(reconstruct))

      runner.train_model(
        train,
        pretty_loss,
        EPOCH_SIZE,
        feed_vars=(input_placeholder, output_placeholder),
        feed_data=pt.train.feed_numpy(BATCH_SIZE, train_data, train_data),
        print_every=None
      )

      accuracy = np.sqrt(runner.evaluate_model(
        pretty_loss,
        TEST_SIZE,
        feed_vars=(input_placeholder, output_placeholder),
        feed_data=pt.train.feed_numpy(BATCH_SIZE, test_data, test_data)))

      print_epoch_info(accuracy, train_epoch, epoch_info, epochs_to_train, epochs_past)
      accuracy_by_epoch.append(accuracy)

      if train_epoch + 1 == epochs_to_train or (train_epoch + 1) % FLAGS.save_every == 0:
        feed = zip(xrange(EPOCH_SIZE), pt.train.feed_numpy(BATCH_SIZE, train_data, train_data))
        encode_batch = lambda data: \
          sess.run([encode_op], dict(zip((input_placeholder, output_placeholder), data)))[0]
        result_batches = [encode_batch(data) for _, data in feed]
        encoding = np.vstack(result_batches)
        checkpoint(runner, sess, encoding, accuracy_by_epoch[-1])

    meta['acu'] = int(np.min(accuracy_by_epoch))
    meta['e'] = int(bookkeeper.global_step().eval() / EPOCH_SIZE)
    ut.reconstruct_images_epochs(np.asarray(epoch_reconstruction),
                                 visualization_data,
                                 save_params=meta,
                                 img_shape=IMAGE_SHAPE)

  ut.print_time('Best Quality: %f for %s' % (np.min(accuracy_by_epoch), ut.to_file_name(meta)))
  return meta, accuracy_by_epoch


def print_epoch_info(accuracy, epoch, epoch_info, epochs, epochs_past):
  previous_epoch_info = '' if epochs_past is None else '+%d' % epochs_past
  ut.print_time('Accuracy after %2d/%d%s epoch %.2f; %s'
                % (epoch + 1,
                   epochs,
                   previous_epoch_info,
                   accuracy,
                   epoch_info)
                )


def search_learning_rate(lrs=[0.01, 0.003, 0.001, 0.0004, 0.0001, 0.00003, 0.00001],
                         enc=encoder_2, dec=decoder_2, epochs=1000):
  result_best, arg_best = None, None
  result_summary = []
  result_list = []
  for lr in lrs:
    meta, accuracy_by_epoch = main(learning_rate=lr, epochs_to_train=epochs, enc=enc, dec=dec)
    result_list.append((ut.to_file_name(meta), accuracy_by_epoch))
    best_accuracy = np.min(accuracy_by_epoch)
    result_summary.append('\n\r lr:%2.5f \tq:%.2f' % (lr, best_accuracy))
    if result_best is None or result_best > best_accuracy:
      result_best = best_accuracy
      arg_best = lr
  meta = {'suf': 'grid_doom_bs', 'e': epochs, 'lrs': lrs, 'enc': enc, 'dec': dec, 'acu': result_best,
          'bs': BATCH_SIZE, 'h': get_layer_info()}
  ut.plot_epoch_progress(meta, result_list)
  print(''.join(result_summary))
  ut.print_info('BEST Q: %d IS ACHIEVED FOR LR: %f' % (result_best, arg_best), 36)


def configure_for_cluster():
  pass


def try_this():
  # identity learning
  pass


if __name__ == '__main__':
  # search_learning_rate()
  # search_learning_rate(lrs=[0.03, 0.001, 0.0001, 0.00001], enc=encoder_2, epochs=50)
  FLAGS.load_state = True
  LAYER_NARROW = 4
  for i in range(10):
    main(learning_rate=0.0004, epochs_to_train=10000)
  # for i in range(10):
  #   search_learning_rate()