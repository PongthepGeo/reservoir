#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import functions as F
#-----------------------------------------------------------------------------------------#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
#-----------------------------------------------------------------------------------------#
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn import metrics
#-----------------------------------------------------------------------------------------#

'''
step 1: input well-log data
'''

data = pd.read_csv('../reservoir_characteristics/datasets/well_logs.csv')
data = data.sort_values(by='Depth', ascending=True)

'''
step 2: encoder. XGBoost needs input labels as an interger with beginning with 0.
'''

le = LabelEncoder()
data['Facies'] = le.fit_transform(data['Facies'])
# print(sorted(data.Facies.unique()))

# NOTE normalizing
drop_cols = ['Facies', 'Formation', 'Well Name', 'Depth'] 
log_names = ['GR', 'ILD_log10',	'DeltaPHI', 'PHIND', 'PE', 'NMM_M', 'RELPOS']
nor_data = F.normalization(data, drop_cols, log_names)
# print('xxx')

# NOTE fill NaN with -999
nor_data.fillna(-999, inplace=True)
miss_data = F.missing_value(nor_data, 'nor_PE')
# print(miss_data)
miss_data.to_csv ('demo.csv', index = None, header=True) 