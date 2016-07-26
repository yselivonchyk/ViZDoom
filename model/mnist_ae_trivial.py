"""MNIST Autoencoder. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np
import utils as ut

import prettytensor as pt
from prettytensor.tutorial import data_utils

tf.app.flags.DEFINE_string(
    'save_path', None, 'Where to save the model checkpoints.')
FLAGS = tf.app.flags.FLAGS

BATCH_SIZE = 20
EPOCH_SIZE = 60000 // BATCH_SIZE
TEST_SIZE = 10000 // BATCH_SIZE
NUM_CLASSES = 2
INPUT_SIZE = 784


tf.app.flags.DEFINE_string('model', 'full',
                           'Choose one of the models, either full or conv')
tf.app.flags.DEFINE_string('input_folder',
                           '../data/circle_basic_1/img/32_32/',
                           'Choose input source folder')
tf.app.flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
tf.app.flags.DEFINE_boolean("interactive", True, "use interactive interface")


FLAGS = tf.app.flags.FLAGS


def encoder(input_tensor):
    return pt.wrap(input_tensor)


def decoder(input_tensor=None, weight_init=tf.truncated_normal):
    return (pt.wrap(input_tensor)
            .fully_connected(
        INPUT_SIZE,
        init=weight_init)).tensor


def loss(input, reconstruciton):
    pretty_input = pt.wrap(input)
    reconstruciton = pt.wrap(reconstruciton).flatten()
    return pretty_input.cross_entropy(reconstruciton)


def main(_=None, weight_init=None, activation_f=tf.nn.sigmoid, data_min=0, data_scale=1.0, epochs=3,learning_rate=None):
    tf.reset_default_graph()
    input_placeholder  = tf.placeholder(tf.float32, [BATCH_SIZE, 2])
    output_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, 28, 28, 1])

    # Grab the data as numpy arrays.
    train_input, train_output = data_utils.mnist(training=True)
    test_input,  test_output  = data_utils.mnist(training=False)

    train_set = ut.mnist_select_n_classes(train_input, train_output, NUM_CLASSES, min=data_min, scale=data_scale)
    test_set  = ut.mnist_select_n_classes(test_input,  test_output,  NUM_CLASSES, min=data_min, scale=data_scale)
    train_input, train_output = train_set[1], train_set[0]
    test_input,  test_output  = test_set[1],  test_set[0]

    ut.print_info('train (min, max): (%f, %f)' % (np.min(train_set[0]), np.max(train_set[0])))

    visual_inputs, visual_output = train_set[1][0:BATCH_SIZE], train_set[0][0:BATCH_SIZE]
    epoch_reconstruction = []

    EPOCH_SIZE = len(train_input) // BATCH_SIZE
    TEST_SIZE = len(test_input) // BATCH_SIZE

    ut.print_info('train: %s' % str(train_input.shape))
    ut.print_info('test:  %s' % str(test_input.shape))
    ut.print_info('output shape:  %s' % str(train_output[0].shape))

    assert visual_inputs.shape == input_placeholder.get_shape()
    assert len(train_input.shape) == len(input_placeholder.get_shape())
    assert len(test_input.shape) == len(input_placeholder.get_shape())
    assert visual_output.shape == output_placeholder.get_shape()
    assert len(train_output.shape) == len(output_placeholder.get_shape())
    assert len(test_output.shape) == len(output_placeholder.get_shape())

    with pt.defaults_scope(activation_fn=activation_f,
                           # batch_normalize=True,
                           # learned_moments_update_rate=0.0003,
                           # variance_epsilon=0.001,
                           # scale_after_normalization=True
                           ):
        with pt.defaults_scope(phase=pt.Phase.train):
            with tf.variable_scope("model") as scope:
                output_tensor = decoder(encoder(input_placeholder), weight_init=weight_init)

    pretty_loss = loss(output_tensor, output_placeholder)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train = pt.apply_optimizer(optimizer, losses=[pretty_loss])

    init = tf.initialize_all_variables()
    runner = pt.train.Runner(save_path=FLAGS.save_path)

    best_q = 100000
    with tf.Session() as sess:
        sess.run(init)
        for epoch in xrange(epochs):
            # Shuffle the training data.

            if epoch % np.ceil(epochs / 40.0) == 0 or epoch + 1 == epochs:
                reconstruct, loss_value = sess.run([output_tensor, pretty_loss], {input_placeholder: visual_inputs, output_placeholder: visual_output})
                epoch_reconstruction.append(reconstruct)
                ut.print_info('epoch:%d (min, max): (%f %f)' %(epoch, np.min(reconstruct), np.max(reconstruct)))

            train_input, train_output = data_utils.permute_data(
                (train_input, train_output))

            runner.train_model(
                train,
                pretty_loss,
                EPOCH_SIZE,
                feed_vars=(input_placeholder, output_placeholder),
                feed_data=pt.train.feed_numpy(BATCH_SIZE, train_input, train_output)
            )
            accuracy = runner.evaluate_model(
                pretty_loss,
                TEST_SIZE,
                feed_vars=(input_placeholder, output_placeholder),
                feed_data=pt.train.feed_numpy(BATCH_SIZE, test_input, test_output))
            ut.print_time('Accuracy after %d epoch %g%%' % (
                epoch + 1, accuracy * 100))
            if best_q > accuracy * 10:
                best_q = accuracy * 10


        ut.reconstruct_images_epochs(np.asarray(epoch_reconstruction), visual_output,
                                     save_params={'suf':'mn_trivs', 'act':activation_f, 'e':epochs, 'opt':optimizer,
                                                  'lr': learning_rate, 'init':weight_init, 'acu': int(best_q)})


# ut.show_plt()

# Boosted version
# w_inits = [None, tf.truncated_normal_initializer(stddev=0.3)]
# l_rates = [0.1]
#
# for w in w_inits:
#     for lr in l_rates:
#         main(epochs=3, weight_init=w, learning_rate=lr)

#


main(epochs=3, weight_init=None, learning_rate=100.0)