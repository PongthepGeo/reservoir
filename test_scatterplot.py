import pandas as tong

xxx = tong.read_csv('../reservoir_characteristics/datasets/well_logs.csv')

import matplotlib.pyplot as plt
import matplotlib.colors as colors
def scatter_plt(x, y, xx, yy, facies, lithocolors, lithofacies, title, ylabel, xlabel, save_file):
	cmap = colors.ListedColormap(lithocolors)
	scatter = plt.scatter(x, y, c=facies, s=30, edgecolors='None', alpha=0.5, cmap=cmap, marker='*')
	scatter = plt.scatter(xx, yy, c=facies, s=30, edgecolors='None', alpha=1, cmap=cmap, marker='X')
	plt.title(title, fontsize=14, fontweight='bold')
	plt.ylabel(ylabel, fontsize=11, fontweight='bold')
	plt.xlabel(xlabel, fontsize=11, fontweight='bold')
	# plt.xlim(0, 350); plt.ylim(0, 80)
	plt.legend(handles=scatter.legend_elements()[0], labels=lithofacies, frameon=True, loc='lower right')
	plt.savefig('./image_out/' + save_file + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
	plt.show()

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
# scatter plot
data = xxx
scatter_plt(data.PE, data.DeltaPHI, data.ILD_log10, data.PHIND, data.Facies, lithocolors, lithofacies, 'Raw Data', 'ILD_log10', 'GR (\u03B3)', 'rawdata')

