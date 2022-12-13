import numpy as np
import matplotlib.pyplot as plt

y = m*x + c
yy = m*pow(x, 2) + d

# NOTE plot
plt.scatter(x ,y)
plt.scatter(x ,yy)
plt.show()