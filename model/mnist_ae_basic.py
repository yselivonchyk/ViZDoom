"""MNIST Autoencoder. One hidden layer size=2, cross entropy"""

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
    'save_path', './tmp/', 'Where to save the model checkpoints.')
FLAGS = tf.app.flags.FLAGS

BATCH_SIZE = 40
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


FLAGS = tf.app.flags.FLAGS


def encoder(input_tensor):
    return (pt.wrap(input_tensor)
            .flatten()
            .fully_connected(NUM_CLASSES * 2)).tensor


def decoder(input_tensor=None):
    return (pt.wrap(input_tensor)
            .fully_connected(INPUT_SIZE)).tensor


def get_reconstruction_cost(output_tensor, target_tensor, epsilon=1e-8):
    target_tensor = (pt.wrap(target_tensor).flatten()).tensor
    return -tf.reduce_sum(output_tensor*tf.log(target_tensor))


def loss(input, reconstruciton):
    pretty_input = pt.wrap(input)
    reconstruciton = pt.wrap(reconstruciton).flatten()
    return pretty_input.cross_entropy(reconstruciton)


def main(_=None):
    image_placeholder  = tf.placeholder(tf.float32, [BATCH_SIZE, 28, 28, 1])
    labels_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, 28, 28, 1])

    # Grab the data as numpy arrays.
    train_images, train_labels = data_utils.mnist(training=True)
    test_images,  test_labels  = data_utils.mnist(training=False)

    train_images, train_labels = ut.mnist_select_n_classes(train_images, train_labels, NUM_CLASSES)
    test_images,  test_labels  = ut.mnist_select_n_classes(test_images, test_labels, NUM_CLASSES)
    train_images *= 2
    test_images *= 2
    visualization_set = train_images[0:BATCH_SIZE]
    epoch_reconstruction = []

    EPOCH_SIZE = len(train_images) // BATCH_SIZE
    TEST_SIZE = len(test_images) // BATCH_SIZE

    ut.print_info('train: %s' % str(train_images.shape))
    ut.print_info('test:  %s' % str(test_images.shape))
    ut.print_info('label example:  %s' % str(train_labels[0]))


    with pt.defaults_scope(activation_fn=tf.nn.tanh,
                           # batch_normalize=True,
                           # learned_moments_update_rate=0.0003,
                           # variance_epsilon=0.001,
                           # scale_after_normalization=True
                           ):
        with pt.defaults_scope(phase=pt.Phase.train):
            with tf.variable_scope("model") as scope:
                output_tensor = decoder(encoder(image_placeholder))

    # rec_loss = get_reconstruction_cost(output_tensor, image_placeholder)
    # pretty_loss = pt.Loss(rec_loss, 'cross entropy loss')
    pretty_loss = loss(output_tensor, labels_placeholder)

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0)
    train = pt.apply_optimizer(optimizer, losses=[pretty_loss])

    init = tf.initialize_all_variables()
    runner = pt.train.Runner(save_path=FLAGS.save_path)

    with tf.Session() as sess:
        sess.run(init)
        for epoch in xrange(7):
            # Shuffle the training data.

            reconstruct, loss_value = sess.run([output_tensor, pretty_loss], {image_placeholder: visualization_set, labels_placeholder: visualization_set})
            epoch_reconstruction.append(reconstruct)
            ut.print_info('epoch:%d (min, max): (%f %f)' %(epoch, np.min(reconstruct), np.max(reconstruct)))

            train_images, train_labels = data_utils.permute_data(
                (train_images, train_labels))

            runner.train_model(
                train,
                pretty_loss,
                EPOCH_SIZE,
                feed_vars=(image_placeholder, labels_placeholder),
                feed_data=pt.train.feed_numpy(BATCH_SIZE, train_images, train_images)
            )
            classification_accuracy = runner.evaluate_model(
                pretty_loss,
                TEST_SIZE,
                feed_vars=(image_placeholder, labels_placeholder),
                feed_data=pt.train.feed_numpy(BATCH_SIZE, test_images, test_images))
            print('Accuracy after %d epoch %g%%' % (
                epoch + 1, classification_accuracy * 100))

    ut.reconstruct_images_epochs(np.asarray(epoch_reconstruction), visualization_set)




if __name__ == '__main__':
    tf.app.run()
