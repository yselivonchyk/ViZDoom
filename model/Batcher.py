import numpy as np


class Batcher:
  data_index = 0
  data_length = 0

  data = None

  def __init__(self, batch_size=None, arrays=None):
    if len(arrays) == 1:
      args = [arrays]
    assert np.std(np.asarray(list(map(len, list(arrays))))) == 0

    self.data_length = len(arrays[0])
    self.batch_size = batch_size
    self.original_data = [print(x.shape) for x in arrays]
    self.original_data = [np.asarray(x) for x in arrays]
    self.shuffle()

  def shuffle(self):
    p = np.random.permutation(np.arange(0, self.data_length))
    # print('_', p)
    # p = np.random.permutation(self.data)
    self.data = [x[p] for x in self.original_data]
    self.data_index = 0

  def get_batch(self, batch_size=None):
    batch_size = batch_size if batch_size is not None else self.batch_size

    reshuffle = self.data_index + batch_size >= self.data_length

    length = batch_size if not reshuffle else self.data_length - self.data_index
    batch = [x[self.data_index: self.data_index + length] for x in self.data]
    self.data_index += length
    if reshuffle:
      self.shuffle()
      part = self.get_batch(batch_size - length)
      batch = [np.concatenate((batch[i], x)) for i, x in enumerate(part)]
    return batch
