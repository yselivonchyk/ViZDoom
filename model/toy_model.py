import tensorflow as tf


x = tf.Variable(0.5)
y = x*x
opt = tf.train.AdagradOptimizer(0.1)
grads = opt.compute_gradients(y)
print(grads)
grad_placeholder = [(tf.placeholder("float", shape=grad[1].get_shape()), grad[1]) for grad in grads]
apply_placeholder_op = opt.apply_gradients(grad_placeholder)


def function1(param):
  return param

def function2(param):
  return param


transform_grads = [(function1(grad[0]), grad[1]) for grad in grads]
apply_transform_op = opt.apply_gradients(transform_grads)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

grad_vals = sess.run([grad[0] for grad in grads])

feed_dict = {}
for i in range(len(grad_placeholder)):
  feed_dict[grad_placeholder[i][0]] = function2(grad_vals[i])
  sess.run(apply_placeholder_op, feed_dict=feed_dict)
  sess.run(apply_transform_op)
