import model.input
import numpy as np
import model.utils as ut
import tensorflow as tf
import model.input as inp
import model.activation_functions as act

def minibatch_test():
    x = np.arange(0, 100, 1)

    inp = model.input.Input(x)
    for j in range(100):
        print(inp.generate_minibatch(17))


def to_file_name_test():
    print(ut.to_file_name({
        'suf': 'test',
        'e': 20,
        'act': tf.nn.sigmoid,
        'opt':tf.train.GradientDescentOptimizer(learning_rate=0.001),
        'lr': 0.001,
        'init': tf.truncated_normal_initializer(stddev=0.35)
    }))
    print(ut.to_file_name({
        'suf': 'test',
        'e': 20,
        'act': tf.nn.sigmoid,
        'opt':tf.train.GradientDescentOptimizer(learning_rate=0.001),
        'lr': 0.001,
        'init': tf.truncated_normal_initializer
    }))


def test_time():
    ut.print_time('one')
    ut.print_time('two')


def test_activation():
    print(act.sigmoid, act.sigmoid.func, act.sigmoid.max, act.sigmoid.min)


def test_ds_scale():
    ds = [-4.0, -2.0, 2.0, 4.0]
    scaled = ut.rescale_ds(ds, -2, -1)
    assert (scaled[0] - scaled[1])*2 == (scaled[1] - scaled[2])
    assert min(scaled) == -2
    assert max(scaled) == -1


test_activation()
test_ds_scale()
# test_time()
# to_file_name_test()
