#-----------------------------------------------------------------------------------------#
import sys
sys.path.append('./Libs') 
import functions as F
#-----------------------------------------------------------------------------------------#
import warnings
warnings.filterwarnings("ignore")
import numpy as np
np.random.seed(1000)
import time as tm
import pandas as pd
#-----------------------------------------------------------------------------------------#
from scipy.signal import medfilt
from keras.utils import np_utils
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
#-----------------------------------------------------------------------------------------#

'''
step 1: import well-log data. For DL, the training data and the validation data are already split.
'''

training_data = pd.read_csv('../datasets/training_data.csv')
blind_data = pd.read_csv('../datasets/nofacies_data.csv')

'''
step 2: prepare data for training, validation and test. DL trains data using the stratified method, which inputs data into neurons by well-log, not splitting by facies.

This experiment will compare adjacent facies and facies.
'''

# NOTE define label colors and names
facies_colors = ['#F4D03F', # Nonmarine sandstone
                 '#F5B041', # Nonmarine coarse siltstone
                 '#DC7633', # Nonmarine fine siltstone
                 '#6E2C00', # Marine siltstone and shale
                 '#1B4F72', # Mudstone (limestone)
                 '#2E86C1', # Wackestone (limestone)
                 '#AED6F1', # Dolomite
                 '#A569BD', # Packstone-grainstone (limestone)
                 '#196F3D'] # Phylloid-algal bafflestone (limestone)
facies_labels = ['SS',
                 'CSiS',
                 'FSiS',
                 'SiSh',
                 'MS',
                 'WS',
                 'D',
                 'PS',
                 'BS']
adjacent_facies = np.array([[1], [0, 2], [1], [4], [3, 5], [4, 6, 7], [5, 7], [5, 6, 8], [6, 7]])
# NOTE assign facies colors 
facies_color_map = {}
for ind, label in enumerate(facies_labels):
    facies_color_map[label] = facies_colors[ind]
# NOTE use a shortman of building function as lambda
training_data.loc[:, 'FaciesLabels'] = training_data.apply(lambda row: F.label_facies(row, facies_labels), axis=1)
# NOTE prepare data by well-log
X = training_data.drop(['Formation', 'Well Name', 'Facies', 'FaciesLabels'], axis=1).values
y = training_data['Facies'].values - 1
X_blind = blind_data.drop(['Formation', 'Well Name'], axis=1).values
wells = training_data['Well Name'].values

'''
step 3: normalized data
'''

scaler = preprocessing.RobustScaler().fit(X)
X_scaled = scaler.transform(X)

'''
step 4: loop training and evaluate the model well-by-well. This technique can help evaluate the inference in each well-log. Bear in mind that some well-logs contain less than nine facies.
'''

# NOTE preallocate memory for timing 
logo = LeaveOneGroupOut()
t0 = tm.time()
f1s_ls = []
acc_ls = []
adj_ls = []
# NOTE begining loops
for train, test in logo.split(X_scaled, y, groups=wells):
    well_name = wells[test[0]]
    X_tr = X_scaled[train]
    X_te = X_scaled[test]
    # NOTE convert y array into categories matrix
    classes = 9
    y_tr = np_utils.to_categorical(y[train], classes)
    # NOTE  call neuron network
    NN = F.DNN()
    # NOTE training
    NN.fit(X_tr, y_tr, epochs=15, batch_size=5, verbose=0) 
    # NOTE predict
    y_hat = np.argmax(NN.predict(X_te), axis=-1)
    y_hat = medfilt(y_hat, kernel_size=7)
    # NOTE condition some cases of model evaluation
    try:
        f1s = f1_score(y[test], y_hat, average='weighted', labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    except:
        f1s = 0
    try:
        conf = confusion_matrix(y[test], y_hat, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        acc = F.accuracy(conf) # similar to f1 micro
    except:
        acc = 0
    try:
        acc_adj = F.accuracy_adjacent(conf, adjacent_facies)
    except:
        acc_adj = 0
    f1s_ls += [f1s]
    acc_ls += [acc]
    adj_ls += [acc_adj]
    print('{:>20s} f1_weigthted:{:.3f} | acc:{:.3f} | acc_adj:{:.3f}'.format(well_name, f1s, acc, acc_adj))
# NOTE print the average for 9 well-logs
t1 = tm.time()
print('Avg F1', np.average(f1s_ls)*100,
      'Avg Acc', np.average(acc_ls)*100,
      'Avg Adj', np.average(adj_ls)*100)
print('Blind Well Test Run Time:', '{:f}'.format((t1-t0)), 'seconds')

'''
step 5: statified K-fold. This step ensures that the splitting data do not significantly enhance or minimize the inference.  
'''

X_train = X_scaled
Y_train = np_utils.to_categorical(y, classes)
t2 = tm.time()
estimator = KerasClassifier(build_fn=F.DNN, nb_epoch=15, batch_size=5, verbose=0)
skf = StratifiedKFold(n_splits=5, shuffle=True)
results_dnn = cross_val_score(estimator, X_train, Y_train, cv= skf.get_n_splits(X_train, Y_train))
print (results_dnn)
t3 = tm.time()
print('Cross Validation Run Time:', '{:f}'.format((t3-t2)), 'seconds')

'''
step 6: use all well-logs for traning and validation.
'''

NN = F.DNN()
NN.fit(X_train, Y_train, epochs=15, batch_size=5, verbose=0)

y_predicted = np.argmax(NN.predict(X_train), axis=-1)
y_predicted = medfilt(y_predicted, kernel_size=7)

f1s = f1_score(y, y_predicted, average='weighted')
Avgf1s = np.average(f1s_ls)*100
print ('f1 training error: ', '{:f}'.format(f1s))
print ('f1 test error: ', '{:f}'.format(Avgf1s))