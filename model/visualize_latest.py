import os
import numpy as np
import visualization as vis

def _get_latest_file(dir="./visualizations/"):
  latest_file, latest_mod_time = None, None
  for root, dirs, files in os.walk(dir):
    # print(root, dirs, files)
    for file in files:
      if '.txt' in file:
        file_path = os.path.join(root, file)
        modification_time = os.path.getmtime(file_path)
        if not latest_mod_time or modification_time > latest_mod_time:
          latest_mod_time = modification_time
          latest_file = file_path
  return latest_file


if __name__ == '__main__':
  latest_file = _get_latest_file()
  data = np.loadtxt(latest_file)
  vis.visualize_encoding(data)
          #
        # layer_info = root.split('_h')[1].split('_')[0]
        # lrate_info = root.split('_lr|')[1].split('_')[0]
        #
        # learning_rate = float(lrate_info)
        # if len(layer_info) < 2:
        #   # print('failed to reconstruct layers', layer_info, file)
        #   continue
        # if learning_rate == 0:
        #   # print('failed to reconstruct learning rate', file)
        #   continue
        # layer_sizes = list(map(int, layer_info.split('|')[1:]))
        # print(learning_rate, layer_sizes)