from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import model.input as inp

import prettytensor as pt
from prettytensor.tutorial import data_utils

tf.app.flags.DEFINE_string(
    'save_path', None, 'Where to save the model checkpoints.')
FLAGS = tf.app.flags.FLAGS

BATCH_SIZE = 40
EPOCH_SIZE = 1000 // BATCH_SIZE
TEST_SIZE = 200 // BATCH_SIZE

tf.app.flags.DEFINE_string('model', 'full',
                           'Choose one of the models, either full or conv')
tf.app.flags.DEFINE_string('input_folder',
                           '../data/circle_basic_1/img/32_32/',
                           'Choose input source folder')

FLAGS = tf.app.flags.FLAGS


def multilayer_fully_connected(images, labels):
    images = pt.wrap(images)
    with pt.defaults_scope(activation_fn=tf.nn.relu, l2loss=0.00001):
        return (images.flatten().fully_connected(100).fully_connected(100)
                .softmax_classifier(10, labels))


def lenet5(images, labels):
    images = pt.wrap(images)
    with pt.defaults_scope(activation_fn=tf.nn.relu, l2loss=0.00001):
        return (images.conv2d(5, 20).max_pool(2, 2).conv2d(5, 50).max_pool(2, 2)
                .flatten().fully_connected(500).softmax_classifier(10, labels))


def main(_=None):
    image_shape = inp.get_image_shape(FLAGS.input_folder)
    batch_shape = (BATCH_SIZE,) + image_shape

    print('>>', image_shape, batch_shape)

    image_placeholder  = tf.placeholder(tf.float32, [BATCH_SIZE, 28, 28, 1])
    labels_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, 10])

    if FLAGS.model == 'full':
        print('fully connected network')
        result = multilayer_fully_connected(image_placeholder, labels_placeholder)
    elif FLAGS.model == 'conv':
        print('conv network')
        result = lenet5(image_placeholder, labels_placeholder)

    accuracy = result.softmax.evaluate_classifier(labels_placeholder,
                                                  phase=pt.Phase.test)

    # Grab the data as numpy arrays.
    train_images, train_labels = data_utils.mnist(training=True)
    test_images,  test_labels  = data_utils.mnist(training=False)

    print(train_images.shape)
    print(train_labels.shape)

    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train_op = pt.apply_optimizer(optimizer, losses=[result.loss])

    runner = pt.train.Runner(save_path=FLAGS.save_path)
    with tf.Session():
        for epoch in xrange(20):
            # Shuffle the training data.
            train_images, train_labels = data_utils.permute_data(
                (train_images, train_labels))
            train_images = inp.get_images(FLAGS.input_folder)

            runner.train_model(
                train_op,
                result.loss,
                EPOCH_SIZE,
                feed_vars=(image_placeholder, labels_placeholder),
                feed_data=pt.train.feed_numpy(BATCH_SIZE, train_images, train_labels),
                print_every=100)
            classification_accuracy = runner.evaluate_model(
                accuracy,
                TEST_SIZE,
                feed_vars=(image_placeholder, labels_placeholder),
                feed_data=pt.train.feed_numpy(BATCH_SIZE, test_images, test_labels))
            print('Accuracy after %d epoch %g%%' % (
                epoch + 1, classification_accuracy * 100))



if __name__ == '__main__':
    tf.app.run()
