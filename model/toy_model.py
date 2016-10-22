import tensorflow as tf
import numpy as np

# x = tf.Variable(0.5)
# y = x*x
# opt = tf.train.AdagradOptimizer(0.1)
# grads = opt.compute_gradients(y)
# print(grads)
# grad_placeholder = [(tf.placeholder("float", shape=grad[1].get_shape()), grad[1]) for grad in grads]
# apply_placeholder_op = opt.apply_gradients(grad_placeholder)
#
#
# def function1(param):
#   return param
#
# def function2(param):
#   return param
#
#
# transform_grads = [(function1(grad[0]), grad[1]) for grad in grads]
# apply_transform_op = opt.apply_gradients(transform_grads)
#
# sess = tf.Session()
# sess.run(tf.initialize_all_variables())
#
# grad_vals = sess.run([grad[0] for grad in grads])
#
# feed_dict = {}
# for i in range(len(grad_placeholder)):
#   feed_dict[grad_placeholder[i][0]] = function2(grad_vals[i])
#   sess.run(apply_placeholder_op, feed_dict=feed_dict)
#   sess.run(apply_transform_op)



#
# @images_to_uint8
# def test(arr, arr2=None):
#   print(1, arr,'\n2', arr2)
#
#
# arr = np.random.randn(2, 2)
# arr2 = np.random.randn(2, 2)
# test(arr, arr2=arr2)
#

x = tf.Variable(99.0)
const = tf.constant(5.0)
x_ = x + tf.stop_gradient(-x) + const # ARGHH
opt = tf.train.MomentumOptimizer(learning_rate=0.0001, momentum=0.9)
train = opt.minimize(x_)

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  print(x_.eval())
  x_original = x.eval()
  sess.run(train)
  grad, = tf.gradients(x_, [x])
  print('grad', grad.eval())
  print(x.eval())
  print(x.eval() - x_original + const.eval())