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

training_data = pd.read_csv('../reservoir_characteristics/datasets/training_data.csv')
blind_data = pd.read_csv('../reservoir_characteristics/datasets/nofacies_data.csv')

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
step: 4 
'''

classes = 9

X_train = X_scaled
Y_train = np_utils.to_categorical(y, classes)

NN = F.DNN()
NN.fit(X_train, Y_train, epochs=15, batch_size=5, verbose=0)

y_predicted = np.argmax(NN.predict(X_train), axis=-1)
y_predicted = medfilt(y_predicted, kernel_size=7)

# NOTE confusion matrices
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt

y_true = training_data.Facies.to_numpy()
dummy = confusion_matrix(y_true, y_predicted)
disp = ConfusionMatrixDisplay(confusion_matrix=dummy)
disp.plot(cmap='Greens') 
plt.savefig('data_out/' + 'test' + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
plt.show()