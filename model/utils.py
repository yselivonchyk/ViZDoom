import time, datetime
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def print_time(string):
    time = datetime.datetime.now().time().strftime('%H:%M:%S')
    res = str(time) + str(string)
    print_color(res)


def print_info(string):
    print_color('\t' +  str(string), color=32)


def print_color(string, color=33):
    res = '%c[1;%dm%s%c[0m' % (27, color, str(string), 27)
    print(res)


def mnist_select_n_classes(train_images, train_labels, num_classes, min=None):
    result_images, result_labels = [], []
    for i, j in zip(train_images, train_labels):
        if np.sum(j[0:num_classes]) > 0:
            result_images.append(i)
            result_labels.append(j[0:num_classes])
    inputs = np.asarray(result_images)

    if min is not None:
        inputs = inputs - np.min(inputs) + min
    return inputs, np.asarray(result_labels)


def _show_picture(pic):
    fig = plt.figure()
    size = fig.get_size_inches()
    fig.set_size_inches(size[0], size[1]*2, forward=True)
    plt.imshow(pic, cmap='Greys_r')
    plt.show()


def reconstruct_image(img, name='image'):
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
    plt.show()


def concat_images(im1, im2, axis=0):
    if im1 is None:
        return im2
    return np.concatenate((im1, im2), axis=axis)


def _reconstruct_picture_line(pictures):
    line_picture = None
    for _, img in enumerate(pictures):
        if len(img.shape) == 1:
            side_len = int(np.sqrt(len(img)))
            img = (np.reshape(img, (side_len, side_len)) * 255)
        if len(img.shape) == 3 and img.shape[2] == 1:
            img = (np.reshape(img, (img.shape[0], img.shape[1])) * 255)
        line_picture = concat_images(line_picture, img)
    return line_picture


def reconstruct_images_epochs(epochs, original=None):
    full_picture = None

    if original is not None:
        min_ref, max_ref = np.min(original), np.max(original)
        print_info('reconstruction char. in epochs (min, max): %f %f' % (np.min(epochs), np.max(epochs)))
        print_info('original values (min, max): %f %f' % (min_ref, max_ref))
        epochs[epochs > max_ref] = max_ref
        epochs[epochs < min_ref] = min_ref
        epochs = (epochs - min_ref) / (max_ref - min_ref)
        original = (original - min_ref) / (max_ref - min_ref)

    for _, epoch in enumerate(epochs):
        full_picture = concat_images(full_picture, _reconstruct_picture_line(epoch), axis=1)
    if original is not None:
        full_picture = concat_images(full_picture, _reconstruct_picture_line(original), axis=1)

    _show_picture(full_picture)
