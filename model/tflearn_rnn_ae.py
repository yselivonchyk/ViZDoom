# -*- coding: utf-8 -*-

""" Auto Encoder Example.
Using an auto encoder on MNIST handwritten digits.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""


from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
import tflearn

import input as inp
import utils as ut
import activation_functions as act
import math


source = "../data/tmp_grey/romb8.2.2/img/"
seq_length = 32


def fetch_datasets():
  activation_func_bounds = act.sigmoid
  original_data, filters = inp.get_images(source)
  original_data = inp.rescale_ds(original_data, activation_func_bounds.min, activation_func_bounds.max)
  part = 1.
  return original_data[:len(original_data)*part],  original_data[len(original_data)*part:]


trainX, testX = fetch_datasets()

image_shape = trainX[0].shape
image_size = np.prod(image_shape)

seq_length = 16
trainX = trainX[:int(len(trainX) / seq_length) * seq_length]
testX = testX[:int(len(testX) / seq_length) * seq_length]

trainX = trainX.reshape((len(trainX)/seq_length, seq_length, image_size))
# testX = testX.reshape((len(testX)/seq_length, seq_length * image_size))

print('train x shape', trainX.shape)
print('output:', np.prod(image_shape))

input = tflearn.input_data([None, seq_length, image_size])
encode = tflearn.lstm(input, 128, dropout=0.8, return_seq=True)
encode = tflearn.lstm(input, 4)
# encode = tflearn.lstm(encode, 6)
# decode = tflearn.lstm(encode, 128, dropout=0.8)
decode = tflearn.fully_connected(encode, image_size * seq_length, activation='sigmoid')

net = tflearn.regression(decode, shuffle_batches=True,
                         optimizer='adam', learning_rate=0.001, loss='mean_square')


model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainX.reshape((len(trainX), seq_length * image_size)),
          # validation_set=(testX, testX.reshape((len(testX), seq_length * image_size))),
          show_metric=True,
          batch_size=4,
          n_epoch=100)


encoding_model = tflearn.DNN(encode, session=model.session)


# Testing the image reconstruction on new data (test set)
image_shape = image_shape if image_shape[-1] != 1 else image_shape[0:-1]
print('new image shape', image_shape)
testX = tflearn.data_utils.shuffle(trainX)[0]
print(testX.shape)
encode_decode = model.predict(testX)
print(np.asarray(encode_decode).shape)
f, a = plt.subplots(2, 10, figsize=(10, 2))
for i in range(10):
  a[0][i].imshow(np.reshape(testX[i, 0], image_shape))
  a[1][i].imshow(np.reshape(encode_decode[i][:image_size], image_shape))
f.show()
plt.draw()
plt.waitforbuttonpress()

exit()
