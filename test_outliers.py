import pandas as tong

xxx = tong.read_csv('../reservoir_characteristics/datasets/well_logs.csv')

def remove_outliers(data, log, min_o, max_o):
	q_low = data[log].quantile(min_o)
	q_hi  = data[log].quantile(max_o)
	return data[(data[log] < q_hi) & (data[log] > q_low)]

# NOTE remove outliers
drop_cols = ['Facies', 'Formation', 'Well Name', 'Depth', 'NM_M', 'RELPOS']
dummy = xxx.drop(drop_cols, axis=1)
log_names     = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']
max_out = [0.995, 0.995, 0.995, 0.995, 0.995] # ordering by wirelines
for count, item in enumerate(log_names):
    y = remove_outliers(dummy, item, 0., max_out[count])
max_out = [0.9, 0.9, 0.9, 0.9, 0.9] # ordering by wirelines
for count, item in enumerate(log_names):
    yy= remove_outliers(dummy, item, 0., max_out[count])
# print(done_data)

import matplotlib.pyplot as plt
plt.scatter(y.GR, y.ILD_log10, s=30, edgecolors='None', alpha=1, c='red', marker='o')
plt.scatter(yy.GR, yy.ILD_log10, s=30, edgecolors='None', alpha=0.5, c='green', marker='*')
# plt.savefig('data_out/' + save_file + '.svg', format='svg', bbox_inches='tight',\
plt.savefig('test.png', format='png', dpi=300, bbox_inches='tight', transparent=True, pad_inches=0.1)
plt.show()

