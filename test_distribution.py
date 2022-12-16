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

# data = pd.read_csv('../datasets/well_logs.csv')
data = pd.read_csv('../reservoir_characteristics/datasets/well_logs.csv')
data = data.sort_values(by='Depth', ascending=True)

'''
step 2: encoder. XGBoost needs input labels as an interger with beginning with 0.
'''

le = LabelEncoder()
data['Facies'] = le.fit_transform(data['Facies'])
# print(sorted(data.Facies.unique()))

'''
step 3: preprocessing
        1. normalization
        2. missing values
'''

# NOTE normalizing
drop_cols = ['Facies', 'Formation', 'Well Name', 'Depth'] 
log_names = ['GR', 'ILD_log10',	'DeltaPHI', 'PHIND', 'PE', 'NMM_M', 'RELPOS']
nor_data = F.normalization(data, drop_cols, log_names)

# NOTE fill NaN with -999
nor_data.fillna(-999, inplace=True)
miss_data = F.missing_value(nor_data, 'nor_PE')
# print(miss_data)
# miss_data.to_csv ('../save_tabular/demo.csv', index = None, header=True) 

'''
step 4: apply encoder at missing values --> we will add one column contained binary data in which indicates data exist and the missing one. 
'''

en_data = F.encoder(miss_data)
en_data.to_csv ('../reservoir_characteristics/save_tabular/demo_en_data.csv', index = None, header=True) 

# '''
# step 5: preparing data for training, validation, and testing. To prove that an inference (trained model) can predict unknown data accurately, geoscientists test the whole well to prove it. In comparison, general ML splits one well into several pieces of facies.
# '''

# NOTE select one well for testing

drop_col_2 = ['NM_M', 'nor_GR', 'nor_ILD_log10','nor_DeltaPHI',	'nor_PHIND',	'nor_PE', 'nor_NMM_M',	'nor_RELPOS', 'miss'] 
en_data = en_data.drop(drop_col_2, axis=1)

selected_well = 'KIMZEY A'
well_names_test = ['SHRIMPLIN', 'ALEXANDER D', 'SHANKLE', 'LUKE G U', 'KIMZEY A', 'CROSS H CATTLE', 'NOLAN', 'Recruit F9', 'NEWBY', 'CHURCHMAN BIBLE']
train = en_data.loc[en_data['Well Name'] != selected_well]
test  = en_data.loc[en_data['Well Name'] == selected_well]
# NOTE training data use some columns for training 
drop_cols = ['Facies', 'Formation', 'Well Name', 'Depth', 'formation_cat'] 
X = train.drop(drop_cols, axis=1) # select training feature 
y = train['Facies'] # select training label
X_test = test.drop(drop_cols, axis=1) # select testing feature
y_test = test['Facies'] # select testing label
# print(X)
# print(y)
dummy = data['Facies'].value_counts().sort_index(ascending=True)
print(dummy)
