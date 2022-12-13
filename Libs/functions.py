import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
#-----------------------------------------------------------------------------------------#
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import preprocessing
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from bokeh.palettes import Set1
from mpl_toolkits.axes_grid1 import make_axes_locatable
#-----------------------------------------------------------------------------------------#
# pip install pigar
# pigar generate
#-----------------------------------------------------------------------------------------#

def ellipse(b, x, a):
    """
	creating simple ellipse function
    """
    p1 = pow(x, 2)/pow(a, 2)
    p2 = np.sqrt(1000 - p1)
    y = b*p2
    return y

def scatter_plot(lithocolors, x, y, facies, lithofacies):
	fig = plt.figure(figsize=(10, 10))
	cmap = colors.ListedColormap(lithocolors)
	scatter = plt.scatter(x, y, c=facies, s=30, edgecolors='black', alpha=1.0, marker='o', cmap=cmap)
	# plt.xlim(0, 350); plt.ylim(0, 80)
	plt.legend(handles=scatter.legend_elements()[0], labels=lithofacies, frameon=True, loc='lower right')
	plt.title('Scatter Plot', fontsize=14, fontweight='bold')
	plt.xlabel('GR', fontsize=11, fontweight='bold')
	plt.ylabel('ILD_log10', fontsize=11, fontweight='bold')
	plt.show()

def compute_kmeans(x, y, number_of_classes):
	'''
	After finished iteration, this function will locate the distances between centroids (number of classes) and each data point.
    '''
	data = np.zeros(shape=(len(x), 2), dtype=float)
	data[:, 0] = x; data[:, 1] = y
	vector_data = data.reshape(-1, 1) 
	random_centroid = 42 # interger number range 0-42
	kmeans = KMeans(n_clusters = number_of_classes, random_state = random_centroid).fit(vector_data)
	kmeans = kmeans.cluster_centers_[kmeans.labels_]
	kmeans = kmeans.reshape(data.shape)
	return kmeans 

def normalized_data(data, lowest_value, highest_value):
    data = (data - data.min()) / (data.max() - data.min())
    return data * (highest_value - lowest_value) + lowest_value

def find_nearest(data):
	'''
	K-means are unsupervised learning so that the return centroid distances need to group the data points into decided classes.
	'''
	x = normalized_data(data[:, 0], 0, 1)
	unique_x = np.unique(x)
	kmean_data = np.zeros(shape=(len(x), 1), dtype=float)
	# NOTE loop x
	for i in range (0, len(x)):
		difference_array = np.absolute(x[i] - unique_x)
		index = difference_array.argmin()
		kmean_data[i, 0] = index
	return kmean_data

def kmeans_plot_centroids(data_x, data_y, facies, number_of_clusters, lithofacies, k_data):
	fig = plt.figure(figsize=(10, 10))
	# NOTE plot kmeans
	scatter = plt.scatter(data_x, data_y, c=facies, s=30, edgecolors='black', alpha=0.4, marker='o', cmap=plt.get_cmap('tab10', number_of_clusters))
	plt.legend(handles=scatter.legend_elements()[0], labels=lithofacies, frameon=True, loc='lower right')
	# NOTE plot centroids
	x, y = np.unique(k_data[:, 0]), np.unique(k_data[:, 1])
	x = x[1:]
	y = np.repeat(y, len(x), axis=0)
	# print(x, y)
	plt.scatter(x, y, c='blue', s=400, edgecolors='black', alpha=1.0, marker='*')
	plt.title('Kmeans with Centroids', fontsize=14, fontweight='bold')
	plt.xlabel('GR', fontsize=11, fontweight='bold')
	plt.ylabel('ILD_log10', fontsize=11, fontweight='bold')
	plt.show()

def encoder(data): # encoder only formation column
	labelencoder = LabelEncoder() # change label to number
	data['formation_cat'] = labelencoder.fit_transform(data['Formation'])
	enc = OneHotEncoder(handle_unknown='ignore')
	enc_df = pd.DataFrame(enc.fit_transform(data[['formation_cat']]).toarray())
	data = data.join(enc_df)
	return data

def normalization(data, drop_cols, log_names):# shift mean to zero
	dummy  = data.drop(drop_cols, axis=1)
	scaler = preprocessing.StandardScaler().fit(dummy)
	dummy  = scaler.transform(dummy)
	for index, i_log_name in enumerate(log_names):
		new_col = 'nor_' + i_log_name
		data.loc[:, str(new_col)] = dummy[:, index]
	return data

def missing_value(data, log_names):
	X = data[log_names].copy()
	X = X.to_numpy()
	dummy = np.zeros(shape=len(X), dtype=np.int8)
	for i in range (0, len(X)):
		if X[i] == -999:
			dummy[i] = 1
	data.loc[:, 'miss'] = dummy
	return data

def difference(logs, pre, label):
	pre = pre[:, 0]
	# dummy = logs[label].to_numpy() - 1 # -1 is bug fixed to compensate number starting from 1 (not 0)
	dummy = logs[label].to_numpy() 
	diff = np.zeros(shape=(len(pre)), dtype=np.int8)
	# print(dummy)
	count = 0
	for i in range (0, len(pre)):
		if dummy[i] - pre[i] == 0:
			diff[i] = 1
			count += 1
	percent_diff = 1 - (count/len(pre))
	percent_diff = round(percent_diff, 4)
	return diff, percent_diff

def custom_metric(logs, label, predictions, save_file):
	lithocolors  = ['#F4D03F', '#F5B041', '#DC7633', '#6E2C00', '#1B4F72', '#2E86C1', '#AED6F1', '#A569BD', '#196F3D']
	logs = logs.sort_values(by='Depth', ascending=True)
	cmap = colors.ListedColormap(lithocolors)
	ztop = logs.Depth.min(); zbot=logs.Depth.max()
	# true = np.repeat(np.expand_dims(logs[label].values-1, 1), 5, 1) 
	true = np.repeat(np.expand_dims(logs[label].values, 1), 5, 1) 
	predictions = np.repeat(np.expand_dims(predictions, 1), 5, 1)
	f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(6, 8))
	ax1.imshow(true, interpolation='none', aspect='auto', cmap=cmap, vmin=0, vmax=8, extent=[1, 5, zbot, ztop])
	ax2.imshow(predictions, interpolation='none', aspect='auto', cmap=cmap, vmin=0, vmax=8)

	diff, percent_diff = difference(logs, predictions, label)
	diff = np.repeat(np.expand_dims(diff, 1), 5, 1)
	cmap = colors.ListedColormap(['black', 'yellow'])
	ax3.imshow(diff, interpolation='none', aspect='auto', cmap=cmap, vmin=0, vmax=1)
	legend_elements = [Patch(facecolor='black', edgecolor='black', label='incorrect'),
					   Patch(facecolor='yellow', edgecolor='black', label='correct')]
	ax3.legend(handles=legend_elements, bbox_to_anchor=(1.9, 1.01), framealpha=1,
				edgecolor='black')	

	for ax in f.get_axes():
		ax.label_outer()
	ax1.set_xticklabels([]); ax2.set_xticklabels([]); ax3.set_xticklabels([])
	ax1.set_xticks([]); ax2.set_xticks([]); ax3.set_xticks([])
	ax1.set_xlabel('True')
	ax2.set_xlabel('Prediction')
	ax3.set_xlabel('Difference')
	ax3.set_title('error: ' + str(percent_diff), loc='center')

	f.suptitle('Well: %s'%logs.iloc[0]['Well Name'], fontsize=14, y=0.94)
	plt.savefig('data_out/' + save_file + '.svg', format='svg', bbox_inches='tight',\
				transparent=True, pad_inches=0.1)
	plt.show()
	# NOTE QC list of facies
	list_true_facies = (logs[label].sort_values(ascending=True)).unique()
	list_pre_facies = np.unique(predictions)
	return percent_diff, list_true_facies, list_pre_facies

def sub_cm(vector):
	dummy = []
	for i in range (0, len(vector)):
		if vector[i] == 0:
			dummy.append('SS')
		elif vector[i] == 1:
			dummy.append('CSiS')
		elif vector[i] == 2:
			dummy.append('FSiS')
		elif vector[i] == 3:
			dummy.append('SiSh')
		elif vector[i] == 4:
			dummy.append('MS')
		elif vector[i] == 5:
			dummy.append('WS')
		elif vector[i] == 6:
			dummy.append('D')
		elif vector[i] == 7:
			dummy.append('PS')
		elif vector[i] == 8:
			dummy.append('BS')
	return dummy

def cm(y_pred, data, selected_well, lithofacies, save_file):
	y_true = data.Facies.loc[data['Well Name'] == selected_well]
	y_true = y_true.to_numpy()
	y_true = sub_cm(y_true)
	y_pred = sub_cm(y_pred)
	weighted_f1 = f1_score(y_true, y_pred, average='weighted')
	dummy = confusion_matrix(y_true, y_pred, labels=lithofacies)
	disp = ConfusionMatrixDisplay(confusion_matrix=dummy, display_labels=lithofacies)
	disp.plot(cmap='Greens') 
	plt.savefig('data_out/' + save_file + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
	plt.show()
	return weighted_f1

def visualize_classifier(model, lithocolors, x, y, lithofacies, labels, title, ylabel, xlabel):
	'''
	For fancy colors please intall: 
	pip install bokeh
	'''
	# TODO scatter plot
	# _, ax = plt.rcParams['figure.figsize'] = (12, 8)
	fig = plt.figure(figsize=(10, 10))
	scatter_colors = colors.ListedColormap(lithocolors)
	scatter = plt.scatter(x, y, c=labels, s=100, cmap=scatter_colors, zorder=3, edgecolors='black', alpha=0.9)
	plt.title(title, fontsize=14, fontweight='bold')
	plt.ylabel(ylabel, fontsize=11, fontweight='bold')
	plt.xlabel(xlabel, fontsize=11, fontweight='bold')
	plt.xlim(30, 140)
	plt.legend(handles=scatter.legend_elements()[0], labels=lithofacies, frameon=True, framealpha=1.0, loc='upper right', fontsize=11)

	# TODO decsion trees	
	X = np.zeros(shape=(len(x), 2), dtype=float)
	X[:, 0] = x; X[:, 1] = y
	model.fit(X, labels)
	xx, yy = np.meshgrid(np.linspace(30, 140, 200), np.linspace(y.min(), y.max(), 200))
	Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
	n_classes = len(np.unique(labels))
	cmap = colors.ListedColormap(Set1[9][0:9]) # use color pallete from bokeh
	plt.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(n_classes + 1) - 0.5, cmap=cmap, zorder=1)
	plt.show()

def donut(data, label_1, environment, lithofacies, litho_colors, save_file):
	dummy = data[label_1].value_counts().sort_index(ascending=True)
	dummy = dummy.to_numpy() # convert pandas to array
	env = np.zeros(shape=3, dtype=np.float)
	env[0] = dummy[:3].sum() # non-marine 
	env[1] = dummy[3] # marine
	env[2] = dummy[4:].sum() # transition 
	# NOTE ring outside
	_, ax = plt.subplots(figsize=(15, 15))
	ax.axis('equal')
	env_colors = ['grey', 'Aqua', '#808000']
	ring_1, _ = ax.pie(env, radius=1.3, colors=env_colors, labels=environment, textprops={'fontsize': 24})
	plt.setp(ring_1, width=0.5, edgecolor='black')
	# NOTE ring inside
	ring_2, _ = ax.pie(dummy, radius=1.3-0.3, labeldistance=0.7, colors=litho_colors, labels=lithofacies, textprops={'fontsize': 24})
	plt.setp(ring_2, width=0.5, edgecolor='black')
	plt.margins(0,0)
	plt.savefig('../image_out/' + save_file + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0.2)
	plt.show()

def log_9_facies(logs, facies_colors, log_colors, log_names, save_file):
	logs = logs.sort_values(by='Depth')
	cmap_facies = colors.ListedColormap(facies_colors[0:len(facies_colors)], 'indexed')
	ztop = logs.Depth.min(); zbot=logs.Depth.max()
	cluster = np.repeat(np.expand_dims(logs['Facies'].values,1), 100, 1)
	f, ax = plt.subplots(nrows=1, ncols=6, figsize=(8, 12))
	for count, item in enumerate(log_names):
		ax[count].plot(logs[item], logs.Depth, color=log_colors[count])
	im = ax[count+1].imshow(cluster, interpolation='none', aspect='auto', cmap=cmap_facies, vmin=1, vmax=9)
	divider = make_axes_locatable(ax[count+1])
	cax = divider.append_axes('right', size='20%', pad=0.05)
	cbar = plt.colorbar(im, cax=cax)
	#! spaces are matter ' SS '
	cbar.set_label((17*' ').join([' SS ', 'CSiS', 'FSiS', 'SiSh', ' MS ', ' WS ', ' D  ', ' PS ', ' BS ']))
	cbar.set_ticks([1, 2, 3, 4, 5, 6, 7, 8, 9]); cbar.set_ticks([])
	cbar.set_ticks([])
	for i in range(len(ax)-1):
		ax[i].set_ylim(ztop, zbot)
		ax[i].invert_yaxis()
		ax[i].grid()
		ax[i].locator_params(axis='x', nbins=3)
	for count, item in enumerate(log_names):
		ax[count].set_xlabel(str(item))
		ax[count].set_xlim(logs[item].min(), logs[item].max()) 
		ax[count+1].set_yticklabels([])
	ax[count+1].set_xlabel('Facies')
	ax[count+1].set_xticklabels([])
	f.suptitle('Well: %s'%logs.iloc[0]['Well Name'], fontsize=14, y=0.92)
	plt.savefig('../image_out/' + save_file + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
	plt.show()

def data_transformation(data, drop_cols, log_names):# shift mean to zero
	dummy = data.drop(drop_cols, axis=1)
	scaler = preprocessing.StandardScaler().fit(dummy)
	dummy = scaler.transform(dummy)
	for count, item in enumerate(log_names):
		data[item] = dummy[:, count]
	return data

def remove_outliers(data, log, min_o, max_o):
	q_low = data[log].quantile(min_o)
	q_hi  = data[log].quantile(max_o)
	return data[(data[log] < q_hi) & (data[log] > q_low)]

def label_facies_1(row, labels, label_col):
	return labels[row[label_col]-1]

def pairplot_scatter(data, facies_labels, label_col, facies_colors, drop_cols, save_file):
	# create new col named Lithofacies to store facies_labels
	data.loc[:, 'lithofacies'] = data.apply(lambda row: label_facies_1(row, facies_labels, label_col), axis=1)
	facies_color_map = {}
	for ind, label in enumerate(facies_labels):
		facies_color_map[label] = facies_colors[ind]
	data = data.drop(drop_cols, axis=1)
	g = sns.PairGrid(
					data, dropna=True, 
					hue='lithofacies', palette=facies_color_map,
					# hue_order=list(reversed(facies_labels)), 
					hue_order=list(facies_labels), 
					)
	g.map_offdiag(sns.scatterplot, edgecolor='gray', marker='X')
	# g.map_offdiag(sns.scatterplot, edgecolor='none', alpha=0.3, sizes=35, markers='x')
	g.map_diag(sns.histplot, multiple="stack", element="step")
	# g.map_offdiag(sns.kdeplot, multiple="stack")
	# g.set(xticks=[-3, 0, 3], yticks=[-, 6, 10])
	g.add_legend()
	# g.fig.subplots_adjust(top=0.9)
	# g.fig.suptitle(title)
	plt.savefig('image_out/' + save_file + '.png', format='png', dpi=300, bbox_inches='tight', transparent=False, pad_inches=0.2)
	plt.show()

def scatter_plt(x, y, facies, lithocolors, lithofacies, title, ylabel, xlabel, save_file):
	cmap = colors.ListedColormap(lithocolors)
	scatter = plt.scatter(x, y, c=facies, s=30, edgecolors='None', alpha=1.0, cmap=cmap, marker='X')
	plt.title(title, fontsize=14, fontweight='bold')
	plt.ylabel(ylabel, fontsize=11, fontweight='bold')
	plt.xlabel(xlabel, fontsize=11, fontweight='bold')
	# plt.xlim(0, 350); plt.ylim(0, 80)
	plt.legend(handles=scatter.legend_elements()[0], labels=lithofacies, frameon=True, loc='lower right')
	plt.savefig('../image_out/' + save_file + '.svg', format='svg', bbox_inches='tight', transparent=True, pad_inches=0)
	plt.show()

############################################################################################################
############################### Maths ML + DL ################################################
############################################################################################################

def MAE(data_y, model):
	sum = 0.
	for i in range (0, len(data_y)):
		sum += abs(data_y[i] - model[i])
	return sum/len(data_y)

def MSE(data_y, model):
	sum = 0.
	for i in range (0, len(data_y)):
		sum += (data_y[i] - model[i])**2
	return sum/len(data_y)

def linear_equation(slope, weights, intercept):
	return slope*weights + intercept

# https://gist.github.com/sagarmainkar/41d135a04d7d3bc4098f0664fe20cf3c
def  cal_cost(theta, X, y):
    '''
    Calculates the cost for given X and Y. The following shows and example of a single dimensional X
    theta = Vector of thetas 
    X     = Row of X's np.zeros((2,j))
    y     = Actual y's np.zeros((2,1))
    where:
        j is the no of features
    '''
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/2*m) * np.sum(np.square(predictions-y))
    return cost

def gradient_descent(X, y, theta, learning_rate=0.01, iterations=100):
    '''
    X    = Matrix of X with added bias units
    y    = Vector of Y
    theta=Vector of thetas np.random.randn(j,1)
    learning_rate 
    iterations = no of iterations
    Returns the final theta vector and array of cost history over no of iterations
    '''
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations,2))
    for it in range(iterations):
        prediction = np.dot(X,theta)
        theta = theta -(1/m)*learning_rate*( X.T.dot((prediction - y)))
        theta_history[it,:] =theta.T
        cost_history[it]  = cal_cost(theta, X, y)
    return theta, cost_history, theta_history

############################################################################################################
############################### Deep Learning Functions ##############################################
############################################################################################################

#-----------------------------------------------------------------------------------------#
from keras.models import Sequential
from keras.constraints import maxnorm
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.constraints import max_norm
#-----------------------------------------------------------------------------------------#

def accuracy(conf):
    total_correct = 0.
    nb_classes = conf.shape[0]
    for i in np.arange(0,nb_classes):
        total_correct += conf[i][i]
    acc = total_correct/sum(sum(conf))
    return acc

def accuracy_adjacent(conf, adjacent_facies):
    nb_classes = conf.shape[0]
    total_correct = 0.
    for i in np.arange(0,nb_classes):
        total_correct += conf[i][i]
        for j in adjacent_facies[i]:
            total_correct += conf[i][j]
    return total_correct / sum(sum(conf))

def label_facies(row, labels):
    return labels[row['Facies'] -1]

def DNN():
    # Model
    model = Sequential()
    model.add(Dense(205, input_dim=8, activation='relu', kernel_constraint=max_norm(5.)))
    model.add(Dense(64, kernel_constraint=max_norm(5.)))
    model.add(Dropout(0.1))
    model.add(Dense(64, kernel_constraint=max_norm(5.)))
    model.add(Dropout(0.1))
    model.add(Dense(69, activation='relu'))
    model.add(Dense(9, activation='softmax'))
    # Compilation
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model