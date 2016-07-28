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

import prettytensor as pt
from prettytensor.tutorial import data_utils

tf.app.flags.DEFINE_string(
    'save_path', None, 'Where to save the model checkpoints.')
FLAGS = tf.app.flags.FLAGS


BATCH_SIZE = 20
EPOCH_SIZE = 60000 // BATCH_SIZE
TEST_SIZE = 10000 // BATCH_SIZE

HIDDEN_0_SIZE = 200
HIDDEN_1_SIZE = 100
layer_info = []     # saves info about layers

IMAGE_SHAPE = None
BATCH_SHAPE = None

tf.app.flags.DEFINE_string('model', 'full',
                           'Choose one of the models, either full or conv')
tf.app.flags.DEFINE_string('input_folder',
                           '../data/tmp/circle_pos_delay_1/img/',
                           'Choose input source folder')
tf.app.flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
tf.app.flags.DEFINE_boolean("interactive", True, "use interactive interface")


FLAGS = tf.app.flags.FLAGS


def encoder(input_tensor):
    global layer_info
    layer_info.append(HIDDEN_0_SIZE)
    return (pt.wrap(input_tensor)
            .flatten()
            .fully_connected(HIDDEN_0_SIZE)
            ).tensor


def encoder_2(input_tensor):
    global layer_info
    layer_info.append(HIDDEN_0_SIZE)
    layer_info.append(HIDDEN_1_SIZE)

    return (pt.wrap(input_tensor)
            .flatten()
            .fully_connected(HIDDEN_1_SIZE)
            .fully_connected(HIDDEN_0_SIZE).tensor)


def encoder_conv(input_tensor):
    global layer_info
    layer_info.append(HIDDEN_0_SIZE)
    layer_info.append(HIDDEN_1_SIZE)

    return (pt.wrap(input_tensor)
            .flatten()
            .fully_connected(HIDDEN_1_SIZE)
            .fully_connected(HIDDEN_0_SIZE).tensor)

def decoder(input_tensor=None, weight_init=tf.truncated_normal):
    return (pt.wrap(input_tensor)
            .fully_connected(
        IMAGE_SHAPE[0]*IMAGE_SHAPE[1]*IMAGE_SHAPE[2],
        init=weight_init))

def get_reconstruction_cost(output_tensor, target_tensor, epsilon=1e-8):
    target_tensor = (pt.wrap(target_tensor).flatten()).tensor
    return -tf.reduce_sum(output_tensor*tf.log(target_tensor))


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
    input_source = np.asarray(inp.get_images(source_folder))
    input_source, _ = data_utils.permute_data((input_source, np.zeros(len(input_source))))

    EPOCH_SIZE = len(input_source) // BATCH_SIZE
    TEST_SIZE  = len(input_source) // BATCH_SIZE

    input_source = ut.rescale_ds(input_source, data_min, data_max)

    return input_source, input_source, input_source[0:BATCH_SIZE]


def get_batch_shape():
    if len(IMAGE_SHAPE) > 2:
        return BATCH_SIZE, IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2]
    else:
        return BATCH_SIZE, IMAGE_SHAPE[0], IMAGE_SHAPE[1], [1]


vizualize_intermidiate = 10


def main(_=None,
         weight_init=None,
         activation=act.sigmoid,
         epochs=5,
         learning_rate=0.01,
         source_folder=FLAGS.input_folder,
         enc=encoder,
         dec=decoder):

    tf.reset_default_graph()

    global IMAGE_SHAPE, layer_info
    best_q, layer_info = None, []
    epoch_reconstruction = []
    IMAGE_SHAPE = inp.get_image_shape(source_folder)

    input_placeholder  = tf.placeholder(tf.float32, get_batch_shape())
    output_placeholder = tf.placeholder(tf.float32, get_batch_shape())

    train_data, test_data, visualization_data = fetch_datasets(activation.min, activation.max, source_folder)
    assert_model(input_placeholder, output_placeholder, test_data, train_data, visualization_data)

    with pt.defaults_scope(activation_fn=activation.func,
                           # batch_normalize=True,
                           # learned_moments_update_rate=0.0003,
                           # variance_epsilon=0.001,
                           # scale_after_normalization=True
                           ):
        with pt.defaults_scope(phase=pt.Phase.train):
            with tf.variable_scope("model") as scope:
                output_tensor = decoder(encoder(input_placeholder), weight_init=weight_init)

    pretty_loss = square_loss(output_tensor, output_placeholder)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train = pt.apply_optimizer(optimizer, losses=[pretty_loss])

    init = tf.initialize_all_variables()
    runner = pt.train.Runner(save_path=FLAGS.save_path)
    with tf.Session() as sess:
        sess.run(init)
        for epoch in xrange(epochs):
            epoch_info = ''

            if epoch % np.ceil(epochs / float(vizualize_intermidiate)) == 0 or epoch + 1 == epochs:
                reconstruct, loss_value = sess.run([output_tensor, pretty_loss], 
                    {input_placeholder: visualization_data, output_placeholder: visualization_data})
                epoch_reconstruction.append(reconstruct)
                epoch_info += 'epoch:%d (min, max): (%f %f)' %(epoch, np.min(reconstruct), np.max(reconstruct))

            runner.train_model(
                train,
                pretty_loss,
                EPOCH_SIZE,
                feed_vars=(input_placeholder, output_placeholder),
                feed_data=pt.train.feed_numpy(BATCH_SIZE, train_data, train_data),
                print_every=None
            )
            accuracy = runner.evaluate_model(
                pretty_loss,
                TEST_SIZE,
                feed_vars=(input_placeholder, output_placeholder),
                feed_data=pt.train.feed_numpy(BATCH_SIZE, test_data, test_data))
            ut.print_time('Accuracy after %2d/%d epoch %.2f; %s' % (epoch + 1, epochs, accuracy, epoch_info))
            if best_q is None or best_q > accuracy:
                best_q = accuracy

        save_params = {'suf': 'doom_bs', 'act': activation.func, 'e': epochs, 'opt': optimizer, 'lr': learning_rate,
                       'init': weight_init, 'acu': int(best_q), 'bs': BATCH_SIZE, 'h': layer_info}
        ut.reconstruct_images_epochs(np.asarray(epoch_reconstruction), visualization_data, save_params=save_params, img_shape=IMAGE_SHAPE)

    ut.print_time('Best Quality: %f for %s' % (best_q, ut.to_file_name(save_params)))
    return best_q


def search_learning_rate(lrs=[100, 5, 0.5, 0.1, 0.01, 0.001, 0.0001], enc=encoder,dec=decoder, epochs=50):
    best_q, best_r = None, None
    res = []
    for lr in lrs:
        q = main(learning_rate=lr, epochs=epochs, enc=enc, dec=dec)
        res.append('\n\r lr:%.4f \tq:%.2f' % (lr, q))
        if best_q is None or best_q > q:
            best_q = q
            best_r = lr
    print(''.join(res))
    ut.print_info('BEST Q: %d IS ACHIEVED FOR LR: %f' %(best_q, best_r), 36)


def try_this():
    # identity learning
    pass

search_learning_rate()
search_learning_rate(enc=encoder_2, epochs=500)

# main(learning_rate=0.001, epochs=5)

# imgs = np.asarray(inp.get_images(FLAGS.input_folder))/255.0
#
# ut.reconstruct_images_epochs(None, imgs[0:10], save_params={'test':'test'}, img_shape=imgs[0].shape)

# current best sigmoid (0, 1) lr=0.001