import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn import preprocessing

def data_transformation(data, drop_cols, log_names):# shift mean to zero
	dummy = data.drop(drop_cols, axis=1)
	scaler = preprocessing.StandardScaler().fit(dummy)
	dummy = scaler.transform(dummy)
	for count, item in enumerate(log_names):
		data[item] = dummy[:, count]
	return data

def remove_outliers(data, log, min_o, max_o):
	q_low = data[log].quantile(min_o)
	q_hi  = data[log].quantile(max_o)
	return data[(data[log] < q_hi) & (data[log] > q_low)]

def label_facies_1(row, labels, label_col):
	return labels[row[label_col]-1]

def pairplot_scatter(data, facies_labels, label_col, facies_colors, drop_cols, save_file):
	data.loc[:, 'lithofacies'] = data.apply(lambda row: label_facies_1(row, facies_labels, label_col), axis=1)
	facies_color_map = {}
	for ind, label in enumerate(facies_labels):
		facies_color_map[label] = facies_colors[ind]
	data = data.drop(drop_cols, axis=1)
	g = sns.PairGrid(
					data, dropna=True, 
					hue='lithofacies', palette=facies_color_map,
					hue_order=list(facies_labels), 
					)
	g.map_offdiag(sns.scatterplot, edgecolor='gray', marker='X')
	g.map_diag(sns.histplot, multiple="stack", element="step")
	g.add_legend()
	plt.savefig('image_out/' + save_file + '.png', format='png', dpi=300, bbox_inches='tight', transparent=False, pad_inches=0.2)
	plt.show()

data = pd.read_csv('../reservoir_characteristics/datasets/well_logs.csv')

lithocolors = ['#F4D03F', # Nonmarine sandstone
               '#F5B041', # Nonmarine coarse siltstone
               '#DC7633', # Nonmarine fine siltstone
               '#6E2C00', # Marine siltstone and shale
               '#1B4F72', # Mudstone (limestone)
               '#2E86C1', # Wackestone (limestone)
               '#AED6F1', # Dolomite
               '#A569BD', # Packstone-grainstone (limestone)
               '#196F3D'] # Phylloid-algal bafflestone (limestone)
lithofacies = ['SS',
               'CSiS',
               'FSiS',
               'SiSh',
               'MS',
               'WS',
               'D',
               'PS',
               'BS']
environment   = ['non-marine', 'marine', 'transition',]  
log_names     = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']
log_colors    = ['green', 'blue', 'grey', 'red', 'black']
drop_cols     = ['Facies', 'Formation', 'Well Name', 'Depth', 'NM_M', 'RELPOS']
label_col     = 'Facies'
selected_well = data.loc[data['Well Name'] == 'CROSS H CATTLE']

# NOTE normalize data
normalized_data = data_transformation(data, drop_cols, log_names)
# print(normalized_data)
# NOTE remove outliers
max_out = [0.995, 0.995, 0.995, 0.995, 0.995] # ordering by wirelines
for count, item in enumerate(log_names):
    normalized_data = remove_outliers(normalized_data, item, 0., max_out[count])
pairplot_scatter(normalized_data, lithofacies, label_col, lithocolors, drop_cols, 'test')