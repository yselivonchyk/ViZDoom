from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
import model.utils as ut
import numpy as np
import time as t


def modify_ds(ds):
    print(ds.images.shape)
    return DataSet(ds.images, ds.labels)


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
mnist = base.Datasets(train=modify_ds(mnist.train), validation=modify_ds(mnist.validation), test=modify_ds(mnist.test))

sess = tf.InteractiveSession()

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
Wh = tf.Variable(tf.zeros([784, 10]))
bh = tf.Variable(tf.zeros([10]))
h = tf.nn.softmax(tf.matmul(x, Wh) + bh)

Wo = tf.Variable(tf.zeros([10, 784]))
bo = tf.Variable(tf.zeros([784]))
y = tf.nn.tanh(tf.matmul(h, Wo) + bo)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 784])
error = tf.reduce_mean(tf.square(y_ - y))
train_step = tf.train.GradientDescentOptimizer(.3).minimize(error)

# Train
tf.initialize_all_variables().run()
for i in range(5000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # reconstruct_image(batch_xs[1])
    train_step.run({x: batch_xs, y_: batch_xs})
    if i % 500 == 0:
        print('batch', i, error.eval({x: batch_xs, y_: batch_xs}))
        print('test ', i, error.eval({x: mnist.test.images, y_: mnist.test.images}))

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
accuracy = tf.reduce_mean(tf.square(y_ - y))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.images}))

res = y.eval({x: mnist.test.images})
ut.reconstruct_images(mnist.test.images[0:50,:], res[0:50,:])