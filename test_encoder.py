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
# data['Facies'] = le.fit_transform(data['Facies'])
# print(sorted(data.Facies.unique()))
data['Formation'] = le.fit_transform(data['Formation'])
# print(sorted(data.Facies.unique()))
print(data['Formation'][0:5])
# print(data.loc[1][0:5])
