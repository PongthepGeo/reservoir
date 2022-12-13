import sys
sys.path.append('./Libs') 
import functions as F
#-----------------------------------------------------------------------------------------#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#-----------------------------------------------------------------------------------------#

'''
step 1: input well-log data
'''

# data = pd.read_csv('../datasets/well_logs.csv')
data = pd.read_csv('../reservoir_characteristics/datasets/well_logs.csv')
# print(data)

'''
step 2: display a pair of well-log
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
# NOTE scatter plot
F.scatter_plot(lithocolors, data['GR'], data['ILD_log10'], data['Facies'], lithofacies)


'''
step 3: compute kmeans and plot their centroids
'''

number_of_clusters = len(data['Facies'].unique()) + 1 # plus 1 for computing in kmean 
k_data = F.compute_kmeans(data['GR'], data['ILD_log10'], number_of_clusters)
facies = F.find_nearest(k_data)
lithofacies = ['facie 1',
               'facie 2',
               'facie 3',
               'facie 4',
               'facie 5',
               'facie 6',
               'facie 7',
               'facie 8',
               'facie 9']
# NOTE plot kmean plus centroids
F.kmeans_plot_centroids(data['GR'], data['ILD_log10'], facies, number_of_clusters, lithofacies, k_data)