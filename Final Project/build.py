'''
Created on May 7, 2015

@author: antonio
'''
import os
import parameters as prm
from utils import *
import cPickle as pickle
from sklearn import svm, ensemble
from sklearn.neighbors import KNeighborsClassifier
import random


def build():
    data_source = prm.params["data_source"].get()
    pickle_directory = prm.params["pickle_directory"].get()
    print("Building CLF...")
    data, labels, rawData = getTrainData_Pickle(data_source)
    
    clf = ensemble.RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=10)
#     clf = svm.SVC(kernel='linear', C=1)
#     clf = KNeighborsClassifier(n_neighbors=20)

    print("Fitting Data...")
    clf.fit(data, labels)
    
    print("Saving CLF...")
    pickle.dump(clf, open(pickle_directory + "clf_" + str(prm.params["chunk_size"].get()) + "_" + data_source +".p", "wb"))        
    
    print("Build Data Done")

