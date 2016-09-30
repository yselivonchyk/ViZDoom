from scipy import misc
import os
import numpy as np
from os import listdir
from os.path import isfile, join
import utils as ut

INPUT_FOLDER = '../data/circle_basic_1/img/32_32'


class Input:
  data = []

  def __init__(self, data):
    self.data = np.asarray(data, dtype=np.float32)
    # normalize
    min, max = np.min(self.data), np.max(self.data)
    self.data = (self.data - min) / max
    self.data_length = len(data)

  def shuffle(self):
    p = np.random.permutation(self.data)
    self.data = self.data[p]
    self.data_index = 0

  data_index = 0
  data_length = 0

  def generate_minibatch(self, batch_size):
    reshuffle = self.data_index + batch_size >= self.data_length

    length = batch_size if not reshuffle else self.data_length - self.data_index
    batch = self.data[self.data_index:self.data_index + length]
    self.data_index += length
    if reshuffle:
      self.shuffle()
      batch = np.concatenate((batch, self.generate_minibatch(batch_size - length)))
    return batch


# def get_image_dimension(folder, channels=False):
#     images = _get_image_file_list(folder)
#     im = Image.open(images[0])
#     return im.size


def _embed_3_axis(img):
  """embed 1 channel greyscale image into [w, h, ch] array"""
  if len(img.shape) == 2:
    res = np.ndarray((img.shape[0], img.shape[1], 1), dtype=np.float32)
    res[:, :, 0] = img
    return res
  return img


def get_image_shape(folder):
  is_combination = _is_combination_of_image_depth(folder)
  if is_combination:
    folder = os.path.join(folder, 'img')
  images = _get_image_file_list(folder)
  image = _embed_3_axis(misc.imread(images[0]))
  shape = list(image.shape)
  if is_combination:
    shape[-1] += 1
  return shape


def get_batch_shape(batch_size, input_folder):
  _image_shape = get_image_shape(input_folder)
  if len(_image_shape) > 2:
    return batch_size, _image_shape[0], _image_shape[1], _image_shape[2]
  else:
    return batch_size, _image_shape[0], _image_shape[1], [1]


def _is_combination_of_image_depth(folder):
  return '/dep' not in folder or '/img' not in folder


def get_images(folder, at_most=None):
  if _is_combination_of_image_depth(folder):
    depth, d_labels = _get_images(os.path.join(folder, 'dep'))
    image, i_labels = _get_images(os.path.join(folder, 'img'))
    assert len(depth) == len(image)
    assert all([d_labels[i] == i_labels[i] for i in range(len(d_labels))])

    shape = list(image.shape)
    shape[-1] += 1
    res = np.ndarray(shape, dtype=image.dtype)
    for i in range(len(depth)):
      res[i,:,:,:3] = image[i]
      res[i,:,:,3] = depth[i,:,:,0]
    return res, i_labels
  else:
    return _get_images(folder, at_most)


def _get_images(folder, at_most=None):
  files = _get_image_file_list(folder)
  files.sort()
  if at_most:
    files = files[0:at_most]
  images = list(map(_embed_3_axis, map(misc.imread, files)))
  to_int = lambda s: int(s.split('/')[-1].split('.')[0][1:])
  labels = list(map(to_int, files))
  ut.print_info('Found %d images: %s...' % (len(labels), str(labels[1:5])))
  images, labels = np.asarray(images), np.asarray(labels, dtype=np.uint32)
  return images, labels


def rescale_ds(ds, min, max):
  ut.print_info('rescale call: (min: %s, max: %s) %d' % (str(min), str(max), len(ds)))
  if max is None:
    return np.asarray(ds) - np.min(ds)
  ds_min, ds_max = np.min(ds), np.max(ds)
  ds_gap = ds_max - ds_min
  scale_factor = (max - min) / ds_gap
  ds = np.asarray(ds) * scale_factor
  shift_factor = min - np.min(ds)
  ds += shift_factor
  return ds


def _get_image_file_list(folder):
  images = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
  images = list(filter(lambda x: '.png' in x or '.jpg' in x, images))
  return images


def get_input_name(input_folder):
  return input_folder.split('/')[-3]


def permute_array(array, random_state=None):
  return permute_data((array,))[0]


def permute_data(arrays, random_state=None):
  """Permute multiple numpy arrays with the same order."""
  if any(len(a) != len(arrays[0]) for a in arrays):
    raise ValueError('All arrays must be the same length.')
  if not random_state:
    random_state = np.random
  order = random_state.permutation(len(arrays[0]))
  return [a[order] for a in arrays]


def permute_array_in_series(array, series_length):
  return permute_data_in_series((array,), series_length)[0]


def permute_data_in_series(arrays, series_length):
  shift_possibilities = len(arrays[0]) % series_length
  series_count = int(len(arrays[0]) / series_length)

  if shift_possibilities == 0:
    shift_possibilities += series_length
    series_count -= 1
  shift = np.random.randint(0, shift_possibilities+1, 1, dtype=np.int32)[0]

  series = np.arange(0, series_count * series_length)\
    .astype(np.int32)\
    .reshape((series_count, series_length))

  series = np.random.permutation(series)
  data_permutation = series.reshape((series_count * series_length))
  data_permutation += shift

  remaining_elements = np.arange(0, len(arrays[0])).astype(np.int32)
  remaining_elements = np.delete(remaining_elements, data_permutation)
  data_permutation = np.concatenate((data_permutation, remaining_elements))

  return[a[data_permutation] for a in arrays]


if __name__ == '__main__':
  print(permute_data_in_series((np.arange(0, 18),), 3))
  exit(0)
  images = get_images(INPUT_FOLDER)

  inputs = Input(images)
  for j in range(10):
    print('adf', inputs.generate_minibatch(7).shape)

  import matplotlib.pyplot as pyplt
  pyplt.imshow(inputs.generate_minibatch(7)[0])
  pyplt.show()
  # face = misc.imread('face.png')
