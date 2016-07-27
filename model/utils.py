import time, datetime
import numpy as np
from matplotlib import pyplot as plt
import os
import collections
import tensorflow as tf

image_folder = './img/'
_start_time = None


def reset_start_time():
    global _start_time
    _start_time = None


def _get_time_offset():
    global _start_time
    time = datetime.datetime.now()
    if _start_time is None:
        _start_time = time
        return '\t\t'
    sec = (time - _start_time).total_seconds()
    return '(+%d)\t' % sec

def print_time(string):
    time = datetime.datetime.now().time().strftime('%H:%M:%S')
    offset = _get_time_offset()
    res = '%s%s %s' % (str(time), offset, str(string))
    print_color(res)


def print_info(string, color=32):
    print_color('\t' + str(string), color=color)


def print_color(string, color=33):
    res = '%c[1;%dm%s%c[0m' % (27, color, str(string), 27)
    print(res)


def mnist_select_n_classes(train_images, train_labels, num_classes, min=None, scale=1.0):
    result_images, result_labels = [], []
    for i, j in zip(train_images, train_labels):
        if np.sum(j[0:num_classes]) > 0:
            result_images.append(i)
            result_labels.append(j[0:num_classes])
    inputs = np.asarray(result_images)

    inputs *= scale
    if min is not None:
        inputs = inputs - np.min(inputs) + min
    return inputs, np.asarray(result_labels)

def rescale_ds(ds, min, max):
    ds_min, ds_max = np.min(ds), np.max(ds)
    ds_gap = ds_max - ds_min
    scale_factor = (max - min)/ds_gap
    ds = np.asarray(ds) * scale_factor
    shift_factor = min - np.min(ds)
    ds += shift_factor
    print('ds', ds.shape, ds)
    return ds


def _save_image(name='image', save_params=None):
    name = name if save_params is None else to_file_name(save_params)
    name = os.path.join(image_folder, name)
    if '.' in name:
        name += '.png'
    plt.savefig(name, dpi=300, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches='tight', pad_inches=0.1,
                frameon=None)


def _show_picture(pic):
    fig = plt.figure()
    size = fig.get_size_inches()
    fig.set_size_inches(size[0], size[1]*2, forward=True)
    plt.imshow(pic, cmap='Greys_r')


def reconstruct_image(img, name='image', save_params=None):
    img = (np.reshape(img, (28, 28)) * 255)
    plt.imshow(img)


def reconstruct_images(input, output, name='recon'):
    res = []
    for _, arr_in in enumerate(input):
        img_in = (np.reshape(arr_in, (28, 28)) * 255)
        img_out = (np.reshape(output[_], (28, 28)) * 255)
        res.append(np.concatenate((img_in, img_out)))
    img = np.zeros((56, 28))
    for i in res:
        img = np.concatenate((img, i), axis=1)
    plt.imshow(img, cmap='Greys_r')


def concat_images(im1, im2, axis=0):
    if im1 is None:
        return im2
    return np.concatenate((im1, im2), axis=axis)


def _reconstruct_picture_line(pictures, shape):
    line_picture = None
    for _, img in enumerate(pictures):
        if len(img.shape) == 1:
            img = (np.reshape(img, shape) * 255)
        if len(img.shape) == 3 and img.shape[2] == 1:
            img = (np.reshape(img, (img.shape[0], img.shape[1])) * 255)
        line_picture = concat_images(line_picture, img)
    return line_picture


def _show_pil(full_picture):
    pass


def show_plt():
    plt.show()


def _construct_img_shape(img):
    assert int(np.sqrt(img.shape[0])) == np.sqrt(img.shape[0])
    return int(np.sqrt(img.shape[0])), int(np.sqrt(img.shape[0])), 1


def reconstruct_images_epochs(epochs, original=None, save_params=None, img_shape=None):
    full_picture = None

    img_shape = img_shape if img_shape is not None else _construct_img_shape(epochs[0][0])
    print(img_shape)

    if original is not None:
        min_ref, max_ref = np.min(original), np.max(original)
        print_info('reconstruction char. in epochs (min, max)|original: (%f %f)|(%f %f)' % (np.min(epochs), np.max(epochs), min_ref, max_ref))
        epochs[epochs > max_ref] = max_ref
        epochs[epochs < min_ref] = min_ref
        epochs = (epochs - min_ref) / (max_ref - min_ref)
        original = (original - min_ref) / (max_ref - min_ref)

    for _, epoch in enumerate(epochs):
        full_picture = concat_images(full_picture, _reconstruct_picture_line(epoch, img_shape), axis=1)
    if original is not None:
        full_picture = concat_images(full_picture, _reconstruct_picture_line(original, img_shape), axis=1)

    _show_picture(full_picture)
    _save_image(save_params=save_params)

    _show_pil(full_picture)


def to_file_name(obj):
    name = ''
    od = collections.OrderedDict(sorted(obj.items()))

    for _, key in enumerate(od):
        value = obj[key]

        if value is None:
            value = 'na'

        if 'function' in str(value):
            value = str(value).split()[1].split('.')[0]
            parts = value.split('_')
            if len(parts) > 1:
                value = ''.join(list(map(lambda x: x.upper()[0], parts)))
        elif ' at ' in str(value):
            value = (str(value).split()[0]).split('.')[-1]
            abbr = [letter for letter in value if letter.isupper()]
            if len(abbr) > 1:
                value = ''.join(abbr)

        if isinstance(value, float):
            value = '%.4f' % value

        if isinstance(value, int):
            value = '%2d' % value

        if len(value) > 8:
            print_info('truncating this: %s %s' % (key, value))
            value = value[0:9]

        if 'suf' in key:
            name = str(value) + name
        else:
            name += '__%s|%s' % (key, str(value))
    return name


def print_model_info():
    print()
    for v in tf.get_collection(tf.GraphKeys.VARIABLES):
        print(v.name, v.get_shape())
    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        print(v.name, v.get_shape())