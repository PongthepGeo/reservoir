import numpy as np
import matplotlib.pyplot as plt

def eq_1(m, x, c):
	y = m*x + c
	return y

def eq_2(m, x, d):
	yy = m*pow(x, 2) + d
	return yy

m = 8
x = np.linspace(0, 10, 50)
c = 2
y = eq_1(m, x, c)
d = 8
yy = eq_2(m, x, d)
# NOTE plot
plt.scatter(x ,y)
plt.scatter(x ,yy)
plt.show()