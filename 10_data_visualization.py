import sys
sys.path.append('./Libs') 
import functions as F
#-----------------------------------------------------------------------------------------#
import numpy as np
import pandas as pd
#-----------------------------------------------------------------------------------------#

'''
step 1: input well-log data
'''

data = pd.read_csv('../datasets/well_logs.csv')

'''
step 2: explore data
		1. statistical data 
		2. null values in each well
		3. number of well and thier names
		4. column names
'''

# print(data.describe())
# NOTE check number of NaN
# null_value_stats = data.isnull().sum(axis=0)
# print(null_value_stats[null_value_stats != 0])
# NOTE check data types
# print(data.dtypes)
# NOTE check well names
# well_names = data['Well Name'].unique()
# print(well_names)
# NOTE check col names
# print(list(data))

'''
step 3: prefine colors and label names
'''

# NOTE define label colors and names
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

'''
step 4: multiplots
		1. plot donut --> environments and lithofacies 
		2. plot well logs --> 5 logs + facies (9 types)
		3. scattered plots + kernel density distributions --> these plots process data using normalization and outlier removal
		4, scattered plots with various conditions
'''

# NOTE donut
# F.donut(data, 'Facies', environment, lithofacies, lithocolors, 'test')
# NOTE log
# F.log_9_facies(selected_well, lithocolors, log_colors, log_names, 'test')
# NOTE normalize data
normalized_data = F.data_transformation(data, drop_cols, log_names)
# print(normalized_data)
# NOTE remove outliers
max_out = [0.995, 0.995, 0.995, 0.995, 0.995] # ordering by wirelines
for count, item in enumerate(log_names):
    normalized_data = F.remove_outliers(normalized_data, item, 0., max_out[count])
# F.pairplot_scatter(normalized_data, lithofacies, label_col, lithocolors, drop_cols, 'test')
# NOTE single scatter plot
F.scatter_plt(data.GR, data.ILD_log10, data.Facies, lithocolors, lithofacies, 'Raw Data', 'ILD_log10', 'GR (\u03B3)', 'rawdata')
# F.scatter_plt(normalized_data.GR, normalized_data.ILD_log10, normalized_data.Facies, lithocolors, lithofacies, 'Normalized Data and Outlier Remover', 'ILD_log10', 'GR (\u03B3)', 'test')
# NOTE fractor (validation) 
normalized_data = normalized_data.sample(frac=.2, random_state=1)
# F.scatter_plt(data.GR, data.PHIND, data.Facies, lithocolors, lithofacies, 'Validating Data (20%)', 'PHIND', 'Gamma (\u03B3)', 'frac_.2')