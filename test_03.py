import pandas as pd

data = pd.read_csv('../reservoir_characteristics/datasets/well_logs.csv')
# print(data.columns)
import matplotlib.pyplot as plt
x = data['GR']
y = data['ILD_log10']
plt.scatter(x, y, c='RdPu_r')
plt.show()
