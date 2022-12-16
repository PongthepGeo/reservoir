#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import functions as F
#-----------------------------------------------------------------------------------------#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna
#-----------------------------------------------------------------------------------------#
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
#-----------------------------------------------------------------------------------------#
# pip install optuna
# pip install plotly
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

'''
step 6: define function for optuna.
'''

def objective(trial, data=X, target=y):
    train_x, test_x, train_y, test_y = train_test_split(data, target,
                                                        test_size=0.33,
                                                        shuffle=y,
                                                        stratify=y)
    param = {
        'booster': 'gbtree',
        'objective': 'reg:squarederror',
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 1e-5, 1e-1),
        'subsample': trial.suggest_float('subsample', 1e-1, 1.),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
        'eta': trial.suggest_float('eta', 1e-5, 1.),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'max_leaves': trial.suggest_int('max_leaves', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300)
    }
    model = xgb.XGBClassifier(**param,
                              num_class=9,
                              eval_metric='mlogloss' # cost function
                              )
    model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=100,verbose=False)
    preds = model.predict(test_x)
    weigthed_f1_score = f1_score(test_y, preds, average='weighted') # evaluation matric
    return weigthed_f1_score

'''
step 7: begin optimization
'''

n_train_iter = 50
sampler = optuna.samplers.TPESampler(seed=42, multivariate=True) 
pruner = optuna.pruners.MedianPruner()
study = optuna.create_study(sampler=sampler, pruner=pruner, direction='minimize')
study.optimize(objective,
               n_trials=n_train_iter,
               gc_after_trial=True)
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)

'''
step 8: plot results 
'''

# fig = optuna.visualization.plot_intermediate_values(study)
# fig.show()
# NOTE plot_optimization_histor: shows the scores from all trials as well as the best score so far at each point.
fig = optuna.visualization.plot_optimization_history(study)
fig.show()
# NOTE plot_parallel_coordinate: interactively visualizes the hyperparameters and scores
fig = optuna.visualization.plot_parallel_coordinate(study)
fig.show()
# NOTE plot_slice: shows the evolution of the search. You can see where in the hyperparameter space your search went and which parts of the space were explored more.
fig = optuna.visualization.plot_slice(study)
fig.show()
# NOTE plot_contour: plots parameter interactions on an interactive chart. You can choose which hyperparameters you would like to explore.
fig = optuna.visualization.plot_contour(study, params=['alpha',
                                                       'min_child_weight', 
                                                       'subsample', 
                                                       'learning_rate', 
                                                       'subsample'
                                                       ])
fig.show()
# NOTE plot parameter imprtances
fig = optuna.visualization.plot_param_importances(study)
fig.show()