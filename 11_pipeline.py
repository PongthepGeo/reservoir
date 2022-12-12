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
# en_data.to_csv ('../reservoir_characteristics/save_tabular/demo_en_data.csv', index = None, header=True) 

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
X = train.drop(drop_cols, axis=1) # select training feature 
y = train['Facies'] # select training label
X_test = test.drop(drop_cols, axis=1) # select testing feature
y_test = test['Facies'] # select testing label
# print(X)
# print(y)
# NOTE separate data for training and testing
X_train, X_val, y_train, y_val = train_test_split(X, y,
						  test_size=0.33,
						  random_state=True,
						  shuffle=y,
						  stratify=y)

'''
step 6: train data and plot learning curves.
'''

# NOTE define parameters for fitting
clf_xgb = xgb.XGBClassifier(booster='gbtree',
                            learning_rate=0.1,
                            objective='multi:softprob',
                            subsample=0.5,
			    max_depth=20,
			    n_estimators=200,
			#     tree_method='gpu_hist',
			#     gpu_id=1,
                            verbosity=1)
# NOTE train
clf_xgb.fit(X_train,
            y_train,
            verbose=True,
            early_stopping_rounds=500,
            eval_metric='merror',
            eval_set=[(X_train, y_train), (X_val, y_val)])

'''
step 7: evaluate an inference on the test well --> ML never sees these data before (prediction).
'''

# NOTE make predictions for the testing well
y_pred = clf_xgb.predict(X_test)
predictions = [round(value) for value in y_pred]
# print(predictions)
# NOTE evaluate predictions
accuracy = accuracy_score(y_test, predictions)
# print(np.unique(y_test))
print(np.unique(predictions))
print("Accuracy: %.2f%%" % (accuracy * 100.0))

'''
step 8: plot learning curves
'''

# NOTE retrieve performance metrics
results = clf_xgb.evals_result()
epochs = len(results['validation_0']['merror'])
x_axis = range(0, epochs)
# NOTE plot log loss
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['merror'], label='train')
ax.plot(x_axis, results['validation_1']['merror'], label='validation')
ax.legend()
plt.ylabel('Multiclass Loss')
plt.title('XGBoost Log Loss')
# NOTE plot classification error
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['merror'], label='train')
ax.plot(x_axis, results['validation_1']['merror'], label='validation')
ax.legend()
plt.ylabel('Classification Error')
plt.title('XGBoost Classification Error')
plt.show()

'''
step 9: plot confusion matrices and custom evaluation 
'''

# NOTE confusion matrices
lithofacies = ['SS', 'CSiS', 'FSiS', 'SiSh', 'MS', 'WS', 'D', 'PS', 'BS']
F.cm(y_pred, en_data, selected_well, lithofacies, 'confusion_matrix')
print(metrics.classification_report(y_test, y_pred))
# NOTE custom evaluation
_, list_true_facies, list_pre_facies = F.custom_metric(test,
                                                       'Facies',
                                                       predictions,
                                                       'test')
print('list true facies: ', list_true_facies)
print('list prediction facies: ', list_pre_facies)

'''
step 10: feature importances
'''

# NOTE feature importance
plt.figure(figsize=(6, 12))
plt.bar(range(len(clf_xgb.feature_importances_)), clf_xgb.feature_importances_)
print(len(clf_xgb.feature_importances_))
drop_cols_2 = ['Facies', 'Formation', 'Well Name', 'Depth', 'formation_cat'] 
new_en_data = en_data.drop(drop_cols_2, axis=1) # select training feature 
labels = new_en_data.columns[:]
x = np.arange(0, len(labels), 1)
plt.xticks(x, labels, rotation=90)
plt.ylabel('values (the more is the better)')
plt.title('Feature Importances', fontweight='bold')
plt.show()