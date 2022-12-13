import numpy as np
import matplotlib.pyplot as plt

model = np.zeros(shape=(300, 300), dtype=np.float)
print(model.shape)

model[125:175, 125:175] = 300

plt.imshow(model)
plt.show()