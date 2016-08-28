from sklearn.manifold import TSNE
import sklearn.manifold as mn
import matplotlib.pyplot as plt
import sklearn.metrics.pairwise as pw
import scipy.spatial.distance as dist
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import utils as ut
import time
import tensorflow as tf


# Next line to silence pyflakes. This import is needed.
Axes3D

# colors = ['grey', 'red', 'magenta']

FLAGS = tf.app.flags.FLAGS
COLOR_MAP = plt.cm.Spectral

def scatter(plot, data, is3d, colors):
  if is3d:
    plot.scatter(data[:, 0], data[:, 1], data[:, 2],
                 marker='.',
                 c=colors,
                 cmap=plt.cm.Spectral,
                 picker=5)
  else:
    plot.scatter(data[:, 0], data[:, 1], c=colors, cmap=plt.cm.Spectral, picker=5)


def print_data_only(data, file_name):
  fig = plt.figure()
  fig.set_size_inches(fig.get_size_inches()[0] * 2, fig.get_size_inches()[1] * 1)

  colors = build_radial_colors(len(data))
  subplot = plt.subplot(111, projection='3d')
  subplot.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors, cmap=COLOR_MAP)
  save_fig(file_name)


def create_gif_from_folder(folder):
  # fig = plt.figure()
  # gif_folder = os.path.join(FLAGS.save_path, 'gif')
  # if not os.path.exists(gif_folder):
  #   os.mkdir(gif_folder)
  # epoch = file_name.split('_e|')[-1].split('_')[0]
  # gif_path = os.path.join(gif_folder, epoch)
  # subplot = plt.subplot(111, projection='3d')
  # subplot.scatter(data[0], data[1], data[2], c=colors, cmap=color_map)
  # save_fig(file_name)
  pass

STD_THRESHOLD = 0.01


def manual_pca(data):
  """remove meaningless dimensions"""
  std = data.std(axis=0)

  # order = np.argsort(std)[::-1]
  order = np.arange(0, data.shape[1]).astype(np.int32)
  std = std[order]
  # filter components by STD but take at least 3

  meaningless = [order[i] for i, x in enumerate(std) if x <= STD_THRESHOLD]
  if any(meaningless):
    ut.print_info('meaningless dimensions on visualization: %s' % str(meaningless))

  order = [order[i] for i, x in enumerate(std) if x > STD_THRESHOLD or i < 3]
  return data[:, order]


def _needs_hessian(manifold):
  if hasattr(manifold, 'dissimilarity') and manifold.dissimilarity == 'precomputed':
    print(True)
    return True
  if hasattr(manifold, 'metric') and manifold.metric == 'precomputed':
    print(True)
    return True
  return False


def visualize_encodings(encodings, file_name=None, grid=None, skip_every=999, fast=False, fig=None):
  hessian_euc = dist.squareform(dist.pdist(encodings, 'euclidean'))
  hessian_cos = dist.squareform(dist.pdist(encodings, 'cosine'))
  encodings = encodings[0:360] if len(encodings) < 1500 else encodings[0:720]
  encodings = manual_pca(encodings)

  if encodings.shape[1] <= 3:
    return print_data_only(encodings, file_name)

  grid = (3, 4) if grid is None else grid
  project_ops = []

  n = 2
  project_ops.append(("LLE ltsa       N:%d" % n, mn.LocallyLinearEmbedding(10, n, method='ltsa')))
  project_ops.append(("LLE modified   N:%d" % n, mn.LocallyLinearEmbedding(10, n, method='modified')))
  project_ops.append(('MDS euclidean  N:%d' % n, mn.MDS(n, max_iter=300, n_init=1, dissimilarity='precomputed')))
  project_ops.append(("TSNE 30/2000   N:%d" % n, TSNE(perplexity=30, n_components=n, init='pca',n_iter=2000)))
  n = 3
  project_ops.append(("LLE ltsa       N:%d" % n, mn.LocallyLinearEmbedding(10, n, method='ltsa')))
  project_ops.append(("LLE modified   N:%d" % n, mn.LocallyLinearEmbedding(10, n, method='modified')))
  project_ops.append(('MDS euclidean  N:%d' % n, mn.MDS(n, max_iter=300, n_init=1, dissimilarity='precomputed')))
  project_ops.append(('MDS cosine     N:%d' % n, mn.MDS(n, max_iter=300, n_init=1, dissimilarity='precomputed')))


  # print(
  #   np.min(hessian_euc),
  #   np.min(hessian_cos),
  #   hessian_euc.size - np.count_nonzero(hessian_euc))

  plot_places = []
  for i in range(12):
    u, v = int(i/(skip_every-1)), i % (skip_every - 1)
    j = v + u * skip_every + 1
    plot_places.append(j)

  fig = fig if fig is not None else plt.figure()
  fig.set_size_inches(fig.get_size_inches()[0] * grid[0] /1.3,
                      fig.get_size_inches()[1] * grid[1]/2.0)

  for i, (name, manifold) in enumerate(project_ops):
    is3d = 'N:3' in name
    if (fast or 'grid' in FLAGS.suffix) and 'MDS' in name:
      continue

    try:
      if is3d: subplot = plt.subplot(grid[0], grid[1], plot_places[i], projection='3d')
      else: subplot = plt.subplot(grid[0], grid[1], plot_places[i])

      data_source = encodings if not _needs_hessian(manifold) else \
        (hessian_cos if 'cosine' in name else hessian_euc)
      projections = manifold.fit_transform(data_source)
      scatter(subplot, projections, is3d, build_radial_colors(len(data_source)))
      subplot.set_title(name)
    except:
      print(name, "Unexpected error: ", sys.exc_info()[0], sys.exc_info()[1] if len(sys.exc_info()) > 1 else '')


  visualize_data_same(encodings, grid=grid, places=plot_places[-4:])
  # visualize_data_same(encodings, grid=grid, places=np.arange(13, 17), dims_as_colors=True)
  # fig.tight_layout()
  # save_fig(file_name)
  ut.print_time('visualization finished')


def save_fig(file_name):
  if not file_name:
    plt.show()
  else:
    plt.savefig(file_name, dpi=300, facecolor='w', edgecolor='w',
                transparent=False, bbox_inches='tight', pad_inches=0.1,
                frameon=None)


def _random_split(sequence, length, original):
  if sequence is None or len(sequence) < length:
    sequence = original.copy()
  sequence = np.random.permutation(sequence)
  return sequence[:length], sequence[length:]


def visualize_data_same(data, grid, places):
  assert len(places) == 4

  all_dimensions = np.arange(0, data.shape[1]).astype(np.int8)
  first_proj, left = _random_split(None, 2, all_dimensions)
  first_color_indexes, _ = _random_split(left, 3, all_dimensions)
  first_color = data_to_colors(data, first_color_indexes)

  second_proj, left = _random_split(left, 2, all_dimensions)
  second_color = build_radial_colors(len(data))

  third_proj, left = _random_split(left, 3, all_dimensions)
  third_color_indexes, _ = _random_split(left, 3, all_dimensions)
  third_color = data_to_colors(data, third_color_indexes)

  forth_proj = np.argsort(data.std(axis=0))[::-1][0:3]
  forth_color = build_radial_colors(len(data))

  for i, (projection, color) in enumerate([
    (first_proj, first_color),
    (second_proj, second_color),
    (third_proj, third_color),
    (forth_proj, forth_color)]
  ):
    points = np.transpose(data[:, projection])

    if len(projection) == 2:
      subplot = plt.subplot(grid[0], grid[1], places[i])
      subplot.scatter(points[0], points[1], c=color, cmap=COLOR_MAP)
    else:
      subplot = plt.subplot(grid[0], grid[1], places[i], projection='3d')
      subplot.scatter(points[0], points[1], points[2], c=color, cmap=COLOR_MAP)
    subplot.set_title('Data %s' % str(projection))


def visualize_data_same_deprecated(data, grid, places, dims_as_colors=False):
  assert len(places) == 4
  dimensions = np.arange(0, np.min([6, data.shape[1]])).astype(np.int)
  assert len(dimensions) == data.shape[1] or len(dimensions) == 6
  projections = [dimensions[x] for x in [[0, 1], [-1, -2], [0, 1, 2], [-1, -2, -3]] ]
  colors = build_radial_colors(len(data))

  for i, dims in enumerate(projections):
    points = np.transpose(data[:, dims])
    if dims_as_colors:
      colors = data_to_colors(np.delete(data.copy(), dims, axis=1))

    if len(dims) == 2:
      subplot = plt.subplot(grid[0], grid[1], places[i])
      subplot.scatter(points[0], points[1], c=colors, cmap=COLOR_MAP)
    else:
      subplot = plt.subplot(grid[0], grid[1], places[i], projection='3d')
      subplot.scatter(points[0], points[1], points[2], c=colors, cmap=COLOR_MAP)
    subplot.set_title('Data %s' % str(dims))


def duplicate_array(array, repeats=None, total_length=None):
  assert repeats is not None or total_length is not None

  if repeats is None:
    repeats = int(np.ceil(total_length/len(array)))
  res = array.copy()
  for i in range(repeats - 1):
    res = np.concatenate((res, array))
  return res if total_length is None else res[:total_length]


def build_radial_colors(length):
  colors = np.arange(0, 180)
  colors = np.concatenate((colors, colors[::-1]))
  colors = duplicate_array(colors, total_length=length)
  return colors


def data_to_colors(data, indexes=None):
  color_data = data[:, indexes] if indexes is not None else data
  shape = color_data.shape

  if shape[1] < 3:
    add = 3 - shape[1]
    add = np.ones((shape[0], add)) * 0.5
    color_data = np.concatenate((color_data, add), axis=1)
  elif shape[1] > 3:
    color_data = color_data[:, 0:3]

  if np.max(color_data) <= 1:
    color_data *= 256
  color_data = color_data.astype(np.int32)
  assert np.mean(color_data) <= 256
  color_data[color_data > 255] = 255
  color_data = color_data * np.asarray([256 ** 2, 256, 1])

  color_data = np.sum(color_data, axis=1)
  color_data = ["#%06x" % c for c in color_data]
  # print('color example', color_data[0])
  return color_data


def visualize_encoding(encodings, folder=None, meta={}, cut=-1):
  # ut.print_info('VISUALIZATION cutting the encodings', color=31)
  # encodings = encodings[1:360]
  file_path = None
  if folder:
    meta['postfix'] = 'pca'
    file_path = ut.to_file_name(meta, folder, 'png')
  visualize_encodings(encodings, file_name=file_path, cut=cut)


def visualize_available_data(root='./', reembed=True, with_mds=False):
  tf.app.flags.DEFINE_string('suffix', 'grid', '')
  FLAGS.suffix = '__' if with_mds else 'grid'
  assert os.path.exists(root)

  files = _list_embedding_files(root, reembed=reembed)
  total_files = len(files)

  for i, file in enumerate(files):
    print('%d/%d %s' % (i + 1, total_files, file))
    data = np.loadtxt(file)
    png_name = file.replace('.txt', '_pca.png')

    visualize_encodings(data, file_name=png_name)
    # folder_info = file.split('/')[-2]
    # layer_info = folder_info.split('_h')[1].split('_')[0]
    # layer_info = '_h|' + layer_info if len(layer_info) >= 2 else ''
    # lrate_info = folder_info.split('_lr|')[1].split('_')[0]
    # epoch_info = folder_info.split('_e|')[1].split('_')[0]
    # png_name = layer_info + '_l|' + lrate_info + '_' + epoch_info + '_' + file[0:-4] + '.png'
    # png_path = os.path.join('./visualizations', png_name)
    #
    # if float(lrate_info) == 0.0004 or float(lrate_info) == 0.0001:
    #   visualize_encodings(data, file_name=png_path)
    #   # visualize_data(data, file_name=png_path[0:-4] + '_data.png')
    # print('%3d/%3d -> %s' % (i, 151, png_path))


def _list_embedding_files(root, reembed=False):
  ecndoding_files = []
  for root, dirs, files in os.walk(root):
    if '/tmp' in root:
      for file in files:
        if '.txt' in file and 'meta' not in file:
          full_path = os.path.join(root, file)
          if not reembed:
            vis_file = full_path.replace('.txt', '_pca.png')
            if os.path.exists(vis_file):
              continue
          ecndoding_files.append(full_path)
  return ecndoding_files


def rerun_embeddings():
  for root, dirs, files in os.walk("./"):
    path = root.split('/')
    if './tmp/' in root and len(path) == 3:
      for file in files:
        if '.txt' in file:
          layer_info = root.split('_h')[1].split('_')[0]
          lrate_info = root.split('_lr|')[1].split('_')[0]

          learning_rate = float(lrate_info)
          if len(layer_info) < 2:
            # print('failed to reconstruct layers', layer_info, file)
            continue
          if learning_rate == 0:
            # print('failed to reconstruct learning rate', file)
            continue
          layer_sizes = list(map(int, layer_info.split('|')[1:]))
          print(learning_rate, layer_sizes)
          #


def print_side_by_side(*args):
  lines, height, width, channels = args[0].shape
  min = 0   #int(args[0].mean())
  print(min)
  print(lines, height)

  stack = args[0]
  if len(args) > 1:
    for i in range(len(args) - 1):
      stack = np.concatenate((stack, args[i+1]), axis=2)
  # stack - array of lines of pictures (arr_0[0], arr_1[0], ...)
  # concatenate lines in one picture (height = tile_h * #lines)
  picture_lines = stack.reshape(lines*height, stack.shape[2], channels)
  picture_lines = np.hstack((
    picture_lines,
    np.ones((lines*height, 2, channels), dtype=np.uint8)*min)) # pad 2 pixels



def stitch_images(*args):
  """Recieves one or many arrays of pictures and stitches them into one picture"""
  lines, height, width, channels = args[0].shape
  min = 0   #int(args[0].mean())
  print('stitch debug:', min, lines, height)

  stack = args[0]
  if len(args) > 1:
    for i in range(len(args) - 1):
      stack = np.concatenate((stack, args[i+1]), axis=2)
  # stack - array of lines of pictures (arr_0[0], arr_1[0], ...)
  # concatenate lines in one picture (height = tile_h * #lines)
  picture_lines = stack.reshape(lines*height, stack.shape[2], channels)
  picture_lines = np.hstack((
    picture_lines,
    np.ones((lines*height, 2, channels), dtype=np.uint8)*min)) # pad 2 pixels

  # slice/reshape to have better image proportions
  return picture_lines, height


def reshape_images(column_picture, height, proportion=1):
  """
  proportion: vertical_size / horizontal size
  """
  lines = int(column_picture.shape[0] / height)
  width = column_picture.shape[1]

  column_size = int(np.ceil(np.sqrt(lines*proportion/height*width)))
  count = int(column_picture.shape[0]/height)
  _, _, channels = column_picture.shape

  picture = column_picture[0:column_size*height, :, :]

  for i in range(int(lines/column_size)):
    start, stop = column_size*height * (i+1), column_size*height * (i+2)
    if start >= len(column_picture):
      break
    if stop < len(column_picture):
      picture = np.hstack((picture, column_picture[start:stop]))
    else:
      last_column = np.vstack((
        column_picture[start:],
        np.ones((stop-len(column_picture), column_picture.shape[1], column_picture.shape[2]),
                dtype=np.uint8)))
      picture = np.hstack((picture, last_column))

  return picture


if __name__ == '__main__':
  visualize_available_data(root='../../VD_backup/All vizdoom data/', reembed=True, with_mds=True)
  # print_data_only(np.random.rand(10, 3), None)
  exit(0)
  # visualize_available_data()
  # rerun_embeddings()
  dec = 123654
  hex = "%06x" % dec
  data = np.random.rand(100, 8)
  data += np.min(data)
  data /= np.max(data)
  print(np.min(data), np.max(data))
  visualize_data_same(data, (2, 2), [1, 2, 3, 4])
  plt.show()