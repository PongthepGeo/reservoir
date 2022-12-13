import sys
sys.path.append('./Libs') 
import functions as F
#-----------------------------------------------------------------------------------------#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
#-----------------------------------------------------------------------------------------#

'''
step 1: input well-log data
'''

# data = pd.read_csv('../datasets/well_logs.csv')
data = pd.read_csv('../reservoir_characteristics/datasets/well_logs.csv')

'''
step 2: plot scatter
'''

# NOTE prepare data
data = data[::10] # select every 10 data points (reduce plotted complexity) 
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
# NOTE select well log to visualize
x = data['GR']
y = data['ILD_log10']
labels = data['Facies']
# NOTE assign label names to match with the lithocolors
# F.scatter_plot(lithocolors, x, y, labels, lithofacies)

'''
step 3: compute decision trees
'''

title = 'Decision Tree (Max Depth: 10)'
ytitle = data.columns[5]
xtitle = data.columns[4]
F.visualize_classifier(DecisionTreeClassifier(max_depth=10), lithocolors,
                        data['GR'], data['ILD_log10'], lithofacies,
                        data['Facies'],
                        title, ytitle, xtitle,
                        )