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
    done_data = remove_outliers(dummy, item, 0., max_out[count])
# print(done_data)

