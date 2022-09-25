

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from skimage import transform
from tabulate import tabulate


def new_data(data,size):
	new_features = np.array(list(map(lambda img: transform.resize(
				img.reshape(8,8),(size,size),mode='constant',preserve_range=True).ravel(),data)))
	return new_features

digits = datasets.load_digits()


# flatten the images
n_samples = len(digits.images)
#data = digits.images.reshape((n_samples, -1))
input_size = 8
data = new_data(digits.data, input_size)

GAMMA = [10,0.1,0.001]
C = [1,10,0.25,5]
best_gam = 0
best_c = 0
best_mean_acc=0
best_train=0
best_val=0
best_test=0
t_l = [['Gamma','C','train acc','dev acc.','val acc']]


for G in GAMMA:
	for c in C:
		hyper_params = {'gamma':G, 'C':c}
		clf = svm.SVC()
		clf.set_params(**hyper_params)
		X_train, X, y_train, y = train_test_split(data, digits.target, test_size=0.1, shuffle=False)
		x_val, x_test, y_val, y_test = train_test_split(X,y,test_size=0.5,shuffle=False)
		clf.fit(X_train, y_train)
		
		#prediction on train, val, test
		predicted_train = clf.predict(X_train)
		predicted_val = clf.predict(x_val)
		predicted_test = clf.predict(x_test)
		
		#accuracy
		acc_val = metrics.accuracy_score(y_val,predicted_val)
		acc_train = metrics.accuracy_score(y_train, predicted_train)
		acc_test = metrics.accuracy_score(y_test, predicted_test)
		
		#append values to the table
		t_l.append([G,c,str(acc_train)+'%',str(acc_val)+'%',str(acc_test)+'%'])
		
		arth_mean_acc =(1/3) * (acc_train+acc_val+acc_test)
		
		if arth_mean_acc>best_mean_acc:
			c_best_gam = G
			c_best_c = c
			c_best_train=acc_train
			c_best_test=acc_test
			c_best_val=acc_val
print(tabulate(t_l, headers='firstrow', tablefmt='fancy_grid'))
print('Best Hyperparameters'+str(c_best_gam)+' and '+str(c_best_c))
print('train, val, and acc '+str(c_best_train)+'%, '+str(c_best_val)+'%, '+str(c_best_test)+'%')
print(" ")

