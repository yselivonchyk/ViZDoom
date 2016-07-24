"""MNIST Autoencoder. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np
import model.utils as ut

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
HIDDEN_0_SIZE = 4
HIDDEN_1_SIZE = 100

tf.app.flags.DEFINE_string('model', 'full',
                           'Choose one of the models, either full or conv')
tf.app.flags.DEFINE_string('input_folder',
                           '../data/circle_basic_1/img/32_32/',
                           'Choose input source folder')
tf.app.flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
tf.app.flags.DEFINE_boolean("interactive", True, "use interactive interface")


FLAGS = tf.app.flags.FLAGS


def encoder(input_tensor):
    return (pt.wrap(input_tensor)
            .flatten()
            .fully_connected(HIDDEN_0_SIZE)).tensor


def encoder_2(input_tensor):
    return (pt.wrap(input_tensor)
            .flatten()
            .fully_connected(HIDDEN_1_SIZE))\
            .fully_connected(HIDDEN_0_SIZE).tensor


def decoder(input_tensor=None, weight_init=tf.truncated_normal):
    return (pt.wrap(input_tensor)
            .fully_connected(
        INPUT_SIZE,
        init=weight_init)).tensor


def get_reconstruction_cost(output_tensor, target_tensor, epsilon=1e-8):
    target_tensor = (pt.wrap(target_tensor).flatten()).tensor
    return -tf.reduce_sum(output_tensor*tf.log(target_tensor))


def loss(input, reconstruciton):
    pretty_input = pt.wrap(input)
    reconstruciton = pt.wrap(reconstruciton).flatten()
    return pretty_input.cross_entropy(reconstruciton)


def assert_model(input_placeholder, output_placeholder, test_input, test_output, train_input, train_output,
                 visual_inputs, visual_output):
    # ut.print_info('train: %s' % str(train_input.shape))
    # ut.print_info('test:  %s' % str(test_input.shape))
    # ut.print_info('output shape:  %s' % str(train_output[0].shape))
    assert visual_inputs.shape == input_placeholder.get_shape()
    assert len(train_input.shape) == len(input_placeholder.get_shape())
    assert len(test_input.shape) == len(input_placeholder.get_shape())
    assert visual_output.shape == output_placeholder.get_shape()
    assert len(train_output.shape) == len(output_placeholder.get_shape())
    assert len(test_output.shape) == len(output_placeholder.get_shape())


def main(_=None, weight_init=tf.random_normal, activation_f=tf.nn.sigmoid, data_min=0, data_scale=1.0, epochs=50,
         learning_rate=0.01, prefix=None):
    tf.reset_default_graph()
    input_placeholder  = tf.placeholder(tf.float32, [BATCH_SIZE, 28, 28, 1])
    output_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, 28, 28, 1])

    # Grab the data as numpy arrays.
    train_input, train_output = data_utils.mnist(training=True)
    test_input,  test_output  = data_utils.mnist(training=False)
    train_set = ut.mnist_select_n_classes(train_input, train_output, NUM_CLASSES, min=data_min, scale=data_scale)
    test_set  = ut.mnist_select_n_classes(test_input,  test_output,  NUM_CLASSES, min=data_min, scale=data_scale)
    train_input, train_output = train_set[0], train_set[0]
    test_input,  test_output  = test_set[0],  test_set[0]
    ut.print_info('train (min, max): (%f, %f)' % (np.min(train_set[0]), np.max(train_set[0])))
    visual_inputs, visual_output = train_set[0][0:BATCH_SIZE], train_set[0][0:BATCH_SIZE]

    epoch_reconstruction = []

    EPOCH_SIZE = len(train_input) // BATCH_SIZE
    TEST_SIZE = len(test_input) // BATCH_SIZE

    assert_model(input_placeholder, output_placeholder, test_input, test_output, train_input, train_output, visual_inputs, visual_output)

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
    train = pt.apply_optimizer(optimizer, losses=[pretty_loss])

    init = tf.initialize_all_variables()
    runner = pt.train.Runner(save_path=FLAGS.save_path)

    best_q = 100000
    with tf.Session() as sess:
        sess.run(init)
        for epoch in xrange(epochs):
            # Shuffle the training data.
            additional_info = ''

            if epoch % np.ceil(epochs / 40.0) == 0 or epoch + 1 == epochs:
                reconstruct, loss_value = sess.run([output_tensor, pretty_loss], {input_placeholder: visual_inputs, output_placeholder: visual_output})
                epoch_reconstruction.append(reconstruct)
                additional_info += 'epoch:%d (min, max): (%f %f)' %(epoch, np.min(reconstruct), np.max(reconstruct))

            train_input, train_output = data_utils.permute_data(
                (train_input, train_output))

            runner.train_model(
                train,
                pretty_loss,
                EPOCH_SIZE,
                feed_vars=(input_placeholder, output_placeholder),
                feed_data=pt.train.feed_numpy(BATCH_SIZE, train_input, train_output),
                print_every=None
            )
            accuracy = runner.evaluate_model(
                pretty_loss,
                TEST_SIZE,
                feed_vars=(input_placeholder, output_placeholder),
                feed_data=pt.train.feed_numpy(BATCH_SIZE, test_input, test_output))
            ut.print_time('Accuracy after %2d/%d epoch %.2f; %s' % (epoch + 1, epochs, accuracy, additional_info))
            if best_q > accuracy:
                best_q = accuracy

        save_params = {'suf': 'mn_basic', 'act': activation_f, 'e': epochs, 'opt': optimizer, 'lr': learning_rate,
                       'init': weight_init, 'acu': int(best_q), 'bs': BATCH_SIZE, 'h': HIDDEN_0_SIZE, 'i':prefix}
        ut.reconstruct_images_epochs(np.asarray(epoch_reconstruction), visual_output, save_params=save_params)

    ut.print_time('Best Quality: %f for %s' % (best_q, ut.to_file_name(save_params)))
    ut.reset_start_time()
    return best_q


def try_this():
    global BATCH_SIZE, HIDDEN_0_SIZE

    # more time for the best configuration
    BATCH_SIZE = 20
    HIDDEN_0_SIZE = 4
    for i in range(3):
        main(learning_rate=0.01, epochs=1000, prefix=i)
    search_learning_rate([100, 10, 3, 1, 0.5, 0.1, 0.01, 0.001], epochs=200)

    # slower learning rate but more epochs might give better converged results
    BATCH_SIZE = 20
    HIDDEN_0_SIZE = 2
    for i in range(3):
        main(learning_rate=0.005, epochs=2000, prefix=i)
    search_learning_rate([100, 10, 3, 1, 0.5, 0.1, 0.01, 0.001], epochs=200)




def search_learning_rate(lrs=[100, 5, 0.5, 0.1, 0.01, 0.001], epochs=None):
    best_q, best_r = None, None
    res = []
    for lr in lrs:
        q = main(learning_rate=lr) if epochs is None else main(learning_rate=lr, epochs=epochs)
        res.append('\n\r lr:%.4f \tq:%.2f' % (lr, q))
        if best_q is None or best_q > q:
            best_q = q
            best_r = lr
    print(''.join(res))
    ut.print_info('BEST Q: %d IS ACHIEVED FOR LR: %f' %(best_q, best_r), 36)

# search_learning_rate()
# search_learning_rate([0.03, 0.02, .015, .01, 0.007, .004])
# main(learning_rate=0.015, epochs=300)
HIDDEN_0_SIZE = 3
main(learning_rate=0.015, epochs=300)

