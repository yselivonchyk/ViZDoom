import tensorflow as tf
import numpy as np
import utils as ut
import input
import DoomModel as dm
import pickle

FLAGS = tf.app.flags.FLAGS


def search_learning_rate(lrs=[0.001, 0.0004, 0.0001, 0.00003, 0.00001],
                         epochs=100):
  FLAGS.suffix = 'grid_lr'
  ut.print_info('START: search_learning_rate', color=31)

  best_result, best_args = None, None
  result_summary, result_list = [], []

  for lr in lrs:
    ut.print_info('STEP: search_learning_rate', color=31)
    FLAGS.learning_rate = lr
    model = dm.DoomModel()
    meta, accuracy_by_epoch = model.train(epochs)
    result_list.append((ut.to_file_name(meta), accuracy_by_epoch))
    best_accuracy = np.min(accuracy_by_epoch)
    result_summary.append('\n\r lr:%2.5f \tq:%.2f' % (lr, best_accuracy))
    if best_result is None or best_result > best_accuracy:
      best_result = best_accuracy
      best_args = lr

  meta = {'suf': 'grid_lr_bs', 'e': epochs, 'lrs': lrs, 'acu': best_result,
          'bs': FLAGS.batch_size, 'h': model.get_layer_info()}
  pickle.dump(result_list, open('search_learning_rate%d.txt' % epochs, "wb"))
  ut.plot_epoch_progress(meta, result_list)
  print(''.join(result_summary))
  ut.print_info('BEST Q: %d IS ACHIEVED FOR LR: %f' % (best_result, best_args), 36)


def search_batch_size(bss=[20, 50, 100], strides=[2, 4, 7], epochs=100):
  FLAGS.suffix = 'grid_bs'
  ut.print_info('START: search_batch_size', color=31)
  best_result, best_args = None, None
  result_summary, result_list = [], []

  print(bss)
  for bs in bss:
    for stride in strides:
      ut.print_info('STEP: search_batch_size %d %d' % (bs, stride), color=31)
      FLAGS.batch_size = bs
      FLAGS.series_length = stride
      model = dm.DoomModel()
      meta, accuracy_by_epoch = model.train(epochs * int(bs / bss[0]))
      result_list.append((ut.to_file_name(meta), accuracy_by_epoch))
      best_accuracy = np.min(accuracy_by_epoch)
      result_summary.append('\n\r bs:%d \tst:%d \tq:%.2f' % (bs, stride, best_accuracy))
      if best_result is None or best_result > best_accuracy:
        best_result = best_accuracy
        best_args = (bs, stride)

  meta = {'suf': 'grid_batch_bs', 'e': epochs, 'acu': best_result, 'ser': stride,
          'bs': FLAGS.batch_size, 'h': model.get_layer_info()}
  pickle.dump(result_list, open('search_batch_size%d.txt' % epochs, "wb"))
  ut.plot_epoch_progress(meta, result_list)
  print(''.join(result_summary))
  ut.print_info('BEST Q: %d IS ACHIEVED FOR bs, st: %d %d' % (best_result, best_args[0], best_args[1]), 36)


"""
h_e	h_n	h_d	q
0100	03	0100	105.00
0100	06	0100	105.93
0100	12	0100	106.05
0100	03	0500	105.73
0100	06	0500	100.66
0100	12	0500	105.31
0100	03	2000	105.81
0100	06	2000	106.00
0100	12	2000	106.04
0500	03	0100	100.30
0500	06	0100	100.61
0500	12	0100	095.50
0500	03	0500	102.53
0500	06	0500	099.28
0500	12	0500	097.00
0500	03	2000	106.50
0500	06	2000	104.44
0500	12	2000	096.58
2000	03	0100	106.28
2000	06	0100	104.38
2000	12	0100	098.60
2000	03	0500	118.14
2000	06	0500	105.19
2000	12	0500	096.39
2000	03	2000	119.36
2000	06	2000	108.37
2000	12	2000	093.44

h_n = 3
h_e / h_d	100	500	2000
100	  105	    105.73	105.81
500	  100.3	  102.53	106.5
2000	106.28	118.14	119.36

h_n = 12
h_e / h_d	100	500	2000
100	  106.05	105.31	106.04
500 	95.5	  97	    96.58
2000	98.6	  96.39	   93.44

h_n = 6
h_e / h_d	100	500	2000
100	  105.93	100.66	106
500	  100.61	99.28	  104.44
2000	104.38	105.19	108.37
"""


def search_layer_sizes(epochs=200):
  FLAGS.suffix = 'grid_h'
  ut.print_info('START: search_layer_sizes', color=31)
  best_result, best_args = None, None
  result_summary, result_list = [], []

  for _, h_encoder in enumerate([100, 500, 2000]):
    for _, h_decoder in enumerate([100, 500, 2000]):
      for _, h_narrow in enumerate([3, 6, 12]):
        model = dm.DoomModel()
        model.layer_encoder = h_encoder
        model.layer_narrow = h_narrow
        model.layer_decoder = h_decoder
        layer_info = str(model.get_layer_info())
        ut.print_info('STEP: search_layer_sizes: ' + str(layer_info), color=31)

        meta, accuracy_by_epoch = model.train(epochs)
        result_list.append((layer_info, accuracy_by_epoch))
        best_accuracy = np.min(accuracy_by_epoch)
        result_summary.append('\n\r h:%s \tq:%.2f' % (layer_info, best_accuracy))
        if best_result is None or best_result > best_accuracy:
          best_result = best_accuracy
          best_args = layer_info

  meta = {'suf': 'grid_H_bs', 'e': epochs, 'acu': best_result,
          'bs': FLAGS.batch_size, 'h': model.get_layer_info()}
  print(''.join(result_summary))
  pickle.dump(result_list, open('search_layer_sizes%d.txt' % epochs, "wb"))
  ut.print_info('BEST Q: %d IS ACHIEVED FOR H: %s' % (best_result, best_args), 36)
  ut.plot_epoch_progress(meta, result_list)


def search_layer_sizes_follow_up():
  """train further 2 best models"""
  FLAGS.save_every = 200
  for i in range(4):
    model = dm.DoomModel()
    model.layer_encoder = 500
    model.layer_narrow = 3
    model.layer_decoder = 100
    model.train(600)

    model = dm.DoomModel()
    model.layer_encoder = 500
    model.layer_narrow = 12
    model.layer_decoder = 500
    model.train(600)


def print_reconstructions_along_with_originals():
  FLAGS.load_from_checkpoint = './tmp/doom_bs__act|sigmoid__bs|20__h|500|5|500__init|na__inp|cbd4__lr|0.0004__opt|AO'
  model = dm.DoomModel()
  files = ut.list_encodings(FLAGS.save_path)
  last_encoding = files[-1]
  print(last_encoding)
  take_only = 20
  data = np.loadtxt(last_encoding)[0:take_only]
  reconstructions = model.decode(data)
  original, _ = input.get_images(FLAGS.input_path, at_most=take_only)
  ut.print_side_by_side(original, reconstructions)


def train_couple_8_models():
  FLAGS.input_path = '../data/tmp/8_pos_delay_3/img/'

  model = dm.DoomModel()
  model.set_layer_sizes([500, 5, 500])
  for i in range(10):
    model.train(1000)

  model = dm.DoomModel()
  model.set_layer_sizes([1000, 10, 1000])
  for i in range(20):
    model.train(1000)


if __name__ == '__main__':
  # print_reconstructions_along_with_originals()
  # train_couple_8_models()
  FLAGS.suffix = 'grid'
  FLAGS.input_path = '../data/tmp/8_pos_delay_3/img/'
  epochs = 10
  search_layer_sizes(epochs=epochs)
  search_batch_size(epochs=epochs)
  FLAGS.batch_size = 40
  search_learning_rate(epochs=epochs)
