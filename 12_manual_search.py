#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import functions as F
#-----------------------------------------------------------------------------------------#
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
#-----------------------------------------------------------------------------------------#
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
#------------------------------------------------------------------------------------------#
import warnings
warnings.filterwarnings('ignore')
#------------------------------------------------------------------------------------------#

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
print(sorted(data.Facies.unique()))

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

'''
step 4: apply encoder at missing values --> we will add one column contained binary data in which indicates data exist and the missing one. 
'''

en_data = F.encoder(miss_data)

'''
step 5: preparing data for training, validation, and testing. To prove that an inference (trained model) can predict unknown data accurately, geoscientists test the whole well to prove it. In comparison, general ML splits one well into several pieces of facies.
'''

# NOTE select one well for testing
selected_well = 'KIMZEY A'
# well_names_test = ['SHRIMPLIN', 'ALEXANDER D', 'SHANKLE', 'LUKE G U', 'KIMZEY A', 'CROSS H CATTLE', 'NOLAN', 'Recruit F9', 'NEWBY', 'CHURCHMAN BIBLE']
train = en_data.loc[en_data['Well Name'] != selected_well]
test  = en_data.loc[en_data['Well Name'] == selected_well]
# NOTE training data use some columns for training 
drop_cols = ['Facies', 'Formation', 'Well Name', 'Depth', 'formation_cat'] 
X = train.drop(drop_cols, axis=1 ) # select training feature 
y = train['Facies'] # select training label
X_test = test.drop(drop_cols, axis=1) # select testing feature
y_test = test['Facies'] # select testing label

# NOTE spliting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# NOTE fit matrix form
dtrain = xgb.DMatrix(data=X_train, label=y_train)
dtest = xgb.DMatrix(data=X_test)

# NOTE define parameters
cost_function = np.zeros(shape=(4, 4), dtype=np.float)
max_depth = [9, 15, 20, 35]
learning_rate = [1, 0.1, 0.15, 0.2]
params = {
          'objective': 'multi:softmax',
          'max_depth': [],
          'learning_rate': [],
          'num_class': 9
         }
for i in range (0, len(max_depth)):
	params['max_depth'] = max_depth[i]
	for ii in range (0, len(learning_rate)):
		params['learning_rate'] = learning_rate[ii]
		# print(params)
		bst = xgb.train(params, dtrain)
		pred = bst.predict(dtest)
		f1_ = f1_score(y_test, pred, average='weighted')
		print('f1: %.5f' % f1_,
        	  'max depth: ', max_depth[i],
              'learning rate: ', learning_rate[ii])
		cost_function[i, ii] = f1_
print(cost_function)

'''
step 6: plot cost function using smooth contours
'''

fig = go.Figure(data = go.Contour(z=cost_function, colorscale='Electric'))
# fig.write_image('pictures/smooth_color.svg', format='svg')
fig.show()
