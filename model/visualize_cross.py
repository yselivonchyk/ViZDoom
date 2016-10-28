"""
Visualize cross-sections of the embedding
"""
import numpy as np
import matplotlib.pyplot as plt
import visualization as vi


def get_figure(fig=None):
  if fig is not None:
    return fig
  fig = plt.figure()
  fig.set_size_inches(fig.get_size_inches()[0] * 2, fig.get_size_inches()[1] * 2)
  return fig


def plot_single(data, select, subplot):
  data = data[:, select]
  subplot.scatter(data[:, 0], data[:, 1], s=15, alpha=1.0,
                  c=vi.build_radial_colors(len(data)),
                  cmap=plt.cm.Spectral)
  data = np.vstack((data, np.asarray([data[0, :]])))
  subplot.plot(data[:, 0], data[:, 1], alpha=0.4)
  subplot.set_xlabel('feature %d' % select[0])
  subplot.set_ylabel('feature %d' % select[1])


def visualize_cross_section(embeddings, fig=None):
  fig = get_figure(fig)
  features = embeddings.shape[-1]
  print(features)
  for i in range(features):
    for j in range(i+1, features):
      size = features - 1
      pos = i*size + j
      print('i, j', i, j, 's, p', size, pos)

      # # x, y, n = i, j, y*features+x+1
      print(size, size, pos)
      subplot = plt.subplot(size, size, pos)
      plot_single(embeddings, [i, j], subplot)


# path = './tmp/run_bs__a|s__bs|30__h|1000|2|10__init|na__inp|8_nt__lr|0.0001__opt|AO__seq|02/encodings__e|07__er|1191.txt'
path = '../../encodings__e|500__z_ac|96.5751.txt'
x = np.loadtxt(path)
# x = np.random.rand(100, 5)
x = vi.manual_pca(x)
x = x[:360]
visualize_cross_section(x, None)
plt.tight_layout()
plt.show()