import datetime
import numpy as np
from matplotlib import pyplot as plt
import os
import collections
import tensorflow as tf
from scipy import misc


IMAGE_FOLDER = './img/'
EPOCH_THRESHOLD = 4

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
    print_info('rescale call: (min: %s, max: %s) %d' % (str(min), str(max), len(ds)))
    if max is None:
        return np.asarray(ds) - np.min(ds)
    ds_min, ds_max = np.min(ds), np.max(ds)
    ds_gap = ds_max - ds_min
    scale_factor = (max - min)/ds_gap
    ds = np.asarray(ds) * scale_factor
    shift_factor = min - np.min(ds)
    ds += shift_factor
    return ds


def _save_image(name='image', save_params=None, image=None):
    if save_params is not None and 'e' in save_params and save_params['e'] < EPOCH_THRESHOLD:
        print_info('IMAGE: output is not saved. epochs %d < %d' % (save_params['e'], EPOCH_THRESHOLD), color=31)
        return

    file_name = name if save_params is None else to_file_name(save_params)
    file_name += '.png'
    name = os.path.join(IMAGE_FOLDER, file_name)

    if image is not None:
        misc.imsave(name, arr=image, format='png')
    #
    # plt.savefig(name, dpi=300, facecolor='w', edgecolor='w',
    #             orientation='portrait', papertype=None, format=None,
    #             transparent=False, bbox_inches='tight', pad_inches=0.1,
    #             frameon=None)


def _show_picture(pic):
    fig = plt.figure()
    size = fig.get_size_inches()
    fig.set_size_inches(size[0], size[1]*2, forward=True)
    plt.imshow(pic, cmap='Greys_r')


def reconstruct_image(img, name='image', save_params=None):
    img = (np.reshape(img, (28, 28)) * 255).astype(np.uint)
    plt.imshow(img)


def reconstruct_images(input, output, name='recon'):
    res = []
    for _, arr_in in enumerate(input):
        img_in = (np.reshape(arr_in, (28, 28)) * 256.0)
        img_out = (np.reshape(output[_], (28, 28)) * 256.0)
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
        img = (img*255).astype(np.uint8)
        if len(img.shape) == 1:
            img = (np.reshape(img, shape))
        if len(img.shape) == 3 and img.shape[2] == 1:
            img = (np.reshape(img, (img.shape[0], img.shape[1])))
        line_picture = concat_images(line_picture, img)
    return line_picture


def show_plt():
    plt.show()


def _construct_img_shape(img):
    assert int(np.sqrt(img.shape[0])) == np.sqrt(img.shape[0])
    return int(np.sqrt(img.shape[0])), int(np.sqrt(img.shape[0])), 1


def reconstruct_images_epochs(epochs, original=None, save_params=None, img_shape=None):
    full_picture = None
    img_shape = img_shape if img_shape is not None else _construct_img_shape(epochs[0][0])

    if original is not None and epochs is not None and len(epochs) >= 3:
        min_ref, max_ref = np.min(original), np.max(original)
        print_info('epoch avg: (original: %s) -> %s' % (str(np.mean(original)), str((np.mean(epochs[0]), np.mean(epochs[1]), np.mean(epochs[2])))))
        print_info('reconstruction char. in epochs (min, max)|original: (%f %f)|(%f %f)' % (np.min(epochs[1:]), np.max(epochs), min_ref, max_ref))

        # epochs[epochs > max_ref] = max_ref
        # epochs[epochs < min_ref] = min_ref
        # epochs = (epochs - min_ref) / (max_ref - min_ref)
        # original = (original - min_ref) / (max_ref - min_ref)

    if epochs is not None:
        for _, epoch in enumerate(epochs):
            full_picture = concat_images(full_picture, _reconstruct_picture_line(epoch, img_shape), axis=1)
    if original is not None:
        # print('original shape', original[0].shape)
        full_picture = concat_images(full_picture, _reconstruct_picture_line(original, img_shape), axis=1)

    _show_picture(full_picture)
    _save_image(save_params=save_params, image=full_picture)


def _abbreviate_string(value):
    abbr = [letter for letter in value if letter.isupper()]
    if len(abbr) > 1:
        return ''.join(abbr)
    return value


def to_file_name(obj, folder=None, ext=None):
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
            value = _abbreviate_string(value)
        elif isinstance(value, type):
            value = _abbreviate_string(value.__name__)

        if isinstance(value, float) or isinstance(value, np.float32):
            value = '%.4f' % value

        if isinstance(value, int):
            value = '%02d' % value

        if isinstance(value, list):
            value = '|'.join(map(str, value))

        if len(value) > 9:
            print_info('truncating this: %s %s' % (key, value))
            value = value[0:9]

        if 'suf' in key:
            name = str(value) + name
        else:
            name += '__%s|%s' % (key, str(value))

    if folder:
        name = os.path.join(folder, name)
    if ext:
        name += '.' + ext
    return name


def print_model_info():
    print()
    for v in tf.get_collection(tf.GraphKeys.VARIABLES):
        print(v.name, v.get_shape())
    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        print(v.name, v.get_shape())


def plot_log(meta, data):
    name = IMAGE_FOLDER + '/grid_lr_' + to_file_name(meta) + '.png'

    x = np.arange(0, len(data[0][1])) + 1
    for _, experiment in enumerate(data):
        plt.semilogy(x, experiment[1], label=experiment[0], marker='o', linestyle='--')
    plt.xlim([1, x[-1]])
    plt.legend(loc='best', fancybox=True, framealpha=0.5, fontsize=8)
    plt.savefig(name, dpi=300, facecolor='w', edgecolor='w',
            transparent=False, bbox_inches='tight', pad_inches=0.1,
            frameon=None)


def configure_folders(FLAGS, meta):
    folder_name = to_file_name(meta) + '/'
    full_path = os.path.join('./tmp/', folder_name)
    if not os.path.exists(full_path):
        os.mkdir(full_path)
    log_path = os.path.join(full_path, 'log')
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    FLAGS.save_path = full_path
    FLAGS.logdir = log_path