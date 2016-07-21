import model.input
import numpy as np

x = np.arange(0, 100, 1)

inp = model.input.Input(x)
for j in range(100):
    print(inp.generate_minibatch(17))

