import matplotlib.pyplot as plt
import seaborn as sns
import pandas as tong


def data_transformation(data, drop_cols, log_names):# shift mean to zero
	from sklearn import preprocessing
	dummy = data.drop(drop_cols, axis=1)
	scaler = preprocessing.StandardScaler().fit(dummy)
	dummy = scaler.transform(dummy)
	for count, item in enumerate(log_names):
		data[item] = dummy[:, count]
	return data

def pairplot_scatter(data, facies_labels, label_col, facies_colors, drop_cols, save_file):
	# create new col named Lithofacies to store facies_labels
	data.loc[:, 'lithofacies'] = data.apply(lambda row: label_facies_1(row, facies_labels, label_col), axis=1)
	facies_color_map = {}
	for ind, label in enumerate(facies_labels):
		facies_color_map[label] = facies_colors[ind]
	data = data.drop(drop_cols, axis=1)
	g = sns.PairGrid(
					data, dropna=True, 
					hue='lithofacies', palette=facies_color_map,
					# hue_order=list(reversed(facies_labels)), 
					hue_order=list(facies_labels), 
					)
	g.map_offdiag(sns.scatterplot, edgecolor='gray', marker='X')
	# g.map_offdiag(sns.scatterplot, edgecolor='none', alpha=0.3, sizes=35, markers='x')
	g.map_diag(sns.histplot, multiple="stack", element="step")
	# g.map_offdiag(sns.kdeplot, multiple="stack")
	# g.set(xticks=[-3, 0, 3], yticks=[-, 6, 10])
	g.add_legend()
	# g.fig.subplots_adjust(top=0.9)
	# g.fig.suptitle(title)
	plt.savefig('image_out/' + save_file + '.png', format='png', dpi=300, bbox_inches='tight', transparent=False, pad_inches=0.2)
	plt.show()

# NOTE import well-logs
xxx = tong.read_csv('../reservoir_characteristics/datasets/well_logs.csv')
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

log_names     = ['GR', 'ILD_log10', 'DeltaPHI', 'PHIND', 'PE']
drop_cols     = ['Facies', 'Formation', 'Well Name', 'Depth', 'NM_M', 'RELPOS']
# normalized_data = data_transformation(xxx, drop_cols, log_names)

testwell = xxx['GR']
plt.plot(testwell)
plt.show()



# pairplot_scatter(normalized_data, lithofacies, 'Facies', lithocolors, drop_cols, 'test')