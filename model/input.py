from scipy import misc
import os
import numpy as np
from os import listdir
from os.path import isfile, join
from PIL import Image



INPUT_FOLDER = '../data/circle_basic_1/img/32_32'


class Input:
    data = []

    def __init__(self, data):
        self.data = np.asarray(data, dtype=np.float32)
        #normalize
        min, max = np.min(self.data), np.max(self.data)
        self.data = (self.data - min)/max
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
        batch = self.data[self.data_index:self.data_index+length]
        self.data_index += length
        if reshuffle:
            self.shuffle()
            batch = np.concatenate((batch, self.generate_minibatch(batch_size -length)))
        return batch


def get_image_dimension(folder, channels=False):
    images = _get_image_file_list(folder)
    im = Image.open(images[0])
    return im.size


def get_image_shape(folder):
    images = _get_image_file_list(folder)
    image = misc.imread(images[0])
    return image.shape


def get_images(folder):
    images = _get_image_file_list(folder)
    images = list(map(misc.imread, images))
    return images


def _get_image_file_list(folder):
    images = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
    images = list(filter(lambda x: '.png' in x or '.jpg' in x, images))
    return images


def main():
    print(get_image_dimension(INPUT_FOLDER))
    images = get_images(INPUT_FOLDER)
    print(len(images))

    inputs = Input(images)
    for j in range(10):
        print(inputs.generate_minibatch(7).shape)

    import matplotlib.pyplot as pyplt
    pyplt.imshow(inputs.generate_minibatch(7)[0])
    pyplt.show()
    # face = misc.imread('face.png')
