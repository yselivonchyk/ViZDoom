import tensorflow as tf
import numpy as np
import utils as ut
import input
import DoomModel as dm

FLAGS = tf.app.flags.FLAGS


def search_learning_rate(lrs=[0.003, 0.001, 0.0004, 0.0001, 0.00003, 0.00001],
                         epochs=20):
  best_result, best_args = None, None
  result_summary, result_list = [], []

  for lr in lrs:
    FLAGS.learning_rate = lr
    model = dm.DoomModel()
    meta, accuracy_by_epoch = model.train(epochs)
    result_list.append((ut.to_file_name(meta), accuracy_by_epoch))
    best_accuracy = np.min(accuracy_by_epoch)
    result_summary.append('\n\r lr:%2.5f \tq:%.2f' % (lr, best_accuracy))
    if best_result is None or best_result > best_accuracy:
      best_result = best_accuracy
      best_args = lr

  meta = {'suf': 'grid_doom_bs', 'e': epochs, 'lrs': lrs, 'acu': best_result,
          'bs': FLAGS.batch_size, 'h': model.get_layer_info()}
  ut.plot_epoch_progress(meta, result_list)
  print(''.join(result_summary))
  ut.print_info('BEST Q: %d IS ACHIEVED FOR LR: %f' % (best_result, best_args), 36)


def search_layer_sizes(epochs=200):
  best_result, best_args = None, None
  result_summary, result_list = [], []

  for _, h_encoder in enumerate[100, 500, 2000]:
    for _, h_decoder in enumerate[100, 500, 2000]:
      for _, h_narrow in enumerate[3, 6, 12]:
        model = dm.DoomModel()
        model.layer_encoder = h_encoder
        model.layer_narrow = h_narrow
        model.layer_decoder = h_decoder
        layer_info = str(model.get_layer_info())

        meta, accuracy_by_epoch = model.train(epochs)
        result_list.append((layer_info, accuracy_by_epoch))
        best_accuracy = np.min(accuracy_by_epoch)
        result_summary.append('\n\r h:%s \tq:%.2f' % (layer_info, best_accuracy))
        if best_result is None or best_result > best_accuracy:
          best_result = best_accuracy
          best_args = layer_info

  meta = {'suf': 'grid_doom_bs', 'e': epochs, 'acu': best_result,
          'bs': FLAGS.batch_size, 'h': model.get_layer_info()}
  print(''.join(result_summary))
  ut.print_info('BEST Q: %d IS ACHIEVED FOR LR: %f' % (best_result, best_args), 36)
  ut.plot_epoch_progress(meta, result_list)


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


if __name__ == '__main__':
    print_reconstructions_along_with_originals()