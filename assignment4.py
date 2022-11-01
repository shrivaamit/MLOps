"""
================================
Recognizing hand-written digits
================================
This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause


#Import required libraries 
import numpy as np
from skimage import transform
from tabulate import tabulate
import matplotlib.pyplot as plt
from statistics import median, pstdev
from sklearn import datasets, svm, tree, metrics
from sklearn.model_selection import train_test_split


def new_data(data,size):
        new_features = np.array(list(map(lambda img: transform.resize(
                                img.reshape(8,8),(size,size),mode='constant',preserve_range=True).ravel(),data)))
        return new_features

digits = datasets.load_digits()
n_samples = len(digits.images)
user_size = 8
count=1
data = new_data(digits.data,user_size)
final_table = [['Category','Test Accuracy for SVMs','Test Accuracy for Decision-Tree-Classifier']]

splits = [0.1,0.2,0.3,0.4,0.5]
for split in splits:
        entry=[]
        print(" ")
        print('Image Size = '+str(user_size)+'x'+str(user_size)+' and Train-Val-Test Split => '+str(int(100*(1-split)))+
                '-'+str(int(50*split))+'-'+str(int(50*split)))

        GAMMA = [1,0.01,0.001,0.1]
        C = [0.5,0.1,1,10]

        best_gam, best_c, best_mean_acc=0, 0, 0
        best_train, best_val, best_test, pred_test = 0, 0, 0, 0
        table = [['Gamma','C','Training Acc.','Val (Dev) Acc.','Test Acc.','Min Acc.','Max Acc.','Median Acc.','Mean Acc.']]
        for GAM in GAMMA:
                for c in C:
                        hyper_params = {'gamma':GAM, 'C':c}
                        clf = svm.SVC()
                        clf.set_params(**hyper_params)
                        X_train, X, y_train, y = train_test_split(data, digits.target, test_size=split, shuffle=False)
                        x_val, x_test, y_val, y_test = train_test_split(X,y,test_size=0.5,shuffle=False)
                        clf.fit(X_train, y_train)
                        predicted_val = clf.predict(x_val)
                        predicted_train = clf.predict(X_train)
                        predicted_test = clf.predict(x_test)
                        val_accuracy = 100*metrics.accuracy_score(y_val,predicted_val)
                        train_accuracy = 100*metrics.accuracy_score(y_train, predicted_train)
                        test_accuracy = 100*metrics.accuracy_score(y_test, predicted_test)
                        mean_acc = (val_accuracy + train_accuracy + test_accuracy)/3
                        min_acc = min([train_accuracy,val_accuracy,test_accuracy])
                        max_acc = max([val_accuracy,train_accuracy,test_accuracy])
                        median_acc = median([val_accuracy,train_accuracy,test_accuracy])
                        table.append([GAM,c,str(train_accuracy)+'%',str(val_accuracy)+'%',str(test_accuracy)+'%',str(min_acc)+'%',
                                        str(max_acc)+'%',str(median_acc)+'%',str(mean_acc)+'%'])
                        if test_accuracy>best_test:
                                best_gam = GAM
                                best_c = c
                                best_train=train_accuracy
                                best_val=val_accuracy
                                best_test=test_accuracy
                                pred_test = predicted_test
        pred_unis, pred_counts = np.unique(pred_test, return_counts=True)
        print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))
        print(" ")
        print('Best Hyperparameters (Gamma and C) => '+str(best_gam)+' and '+str(best_c))
        print('Train, Val (Dev) and Test Accuracies => '+str(best_train)+'%, '+str(best_val)+'%, '+str(best_test)+'%')
        print(" ")
        entry.append(count)
        entry.append(best_test)
        max_depth = [20,40, 60, 120]
        max_leaf = [250,500,700, 1050]

        best_depth, best_leaf = 0, 0
        best_mean_accd=0
        best_traind, best_testd, best_vald, pred_testd = 0, 0, 0, 0
        table_ii = [['Max_Depth','Max_Leaf_Nodes','Training Acc.','Val (Dev) Acc.','Test Acc.','Min Acc.','Max Acc.','Median Acc.','Mean Acc.']]
        for dep in max_depth:
                for leaf in max_leaf:
                        hyper_paramsd = {'max_depth':dep, 'max_leaf_nodes':leaf}
                        clfd = tree.DecisionTreeClassifier()
                        clfd.set_params(**hyper_paramsd)
                        X_train, X, y_train, y = train_test_split(data, digits.target, test_size=split, shuffle=False)
                        x_val, x_test, y_val, y_test = train_test_split(X,y,test_size=0.5,shuffle=False)
                        clfd.fit(X_train, y_train)
                        predicted_val = clfd.predict(x_val)
                        predicted_train = clfd.predict(X_train)
                        predicted_test = clfd.predict(x_test)
                        val_accuracy = 100*metrics.accuracy_score(y_val,predicted_val)
                        train_accuracy = 100*metrics.accuracy_score(y_train, predicted_train)
                        test_accuracy = 100*metrics.accuracy_score(y_test, predicted_test)
                        mean_acc = (val_accuracy + train_accuracy + test_accuracy)/3
                        min_acc = min([train_accuracy,val_accuracy,test_accuracy])
                        max_acc = max([val_accuracy,train_accuracy,test_accuracy])
                        median_acc = median([val_accuracy,train_accuracy,test_accuracy])
                        table_ii.append([dep,leaf,str(train_accuracy)+'%',str(val_accuracy)+'%',str(test_accuracy)+'%',str(min_acc)+'%',
                                        str(max_acc)+'%',str(median_acc)+'%',str(mean_acc)+'%'])
                        if test_accuracy>best_testd:
                                best_depth = dep
                                best_leaf = leaf
                                best_traind=train_accuracy
                                best_vald=val_accuracy
                                best_testd=test_accuracy
                                pred_testd = predicted_test
        pred_uniqued, pred_countd = np.unique(pred_testd, return_counts=True)
        print(tabulate(table_ii, headers='firstrow', tablefmt='fancy_grid'))
        entry.append(best_testd)
        print('Train, Val (Dev) and Test Accuracies => '+str(best_traind)+'%, '+str(best_vald)+'%, '+str(best_testd)+'%')
        print(" ")
        final_table.append(entry)
        print("Best SVM:", dict(zip(pred_unis, pred_counts)))
        print("Best DecisionTreeClf:", dict(zip(pred_uniqued, pred_countd)))

svm_list = [int(final_table[i][1]) for i in range(1,5)]
dtc_list = [int(final_table[i][2]) for i in range(1,5)]
mean_svm = sum(svm_list)/len(svm_list)
mean_dtc = sum(dtc_list)/len(dtc_list)
std_svm = pstdev(svm_list)
std_dtc = pstdev(dtc_list)
final_table.append(['mean',mean_svm,mean_dtc])
final_table.append(['std',std_svm,std_dtc])
print(tabulate(final_table, headers='firstrow', tablefmt='fancy_grid'))


