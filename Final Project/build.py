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


def build(source="test_protocol"):
    pickle_directory = prm.params["pickle_directory"].get()
    print("Building CLF...")
    if source == "test_protocol":
        data, labels, rawData = getTrainData_Pickle("test_protocol")
    else:
        data, labels, rawData = getTrainData_Pickle("voxforge")
    
    clf = ensemble.RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=10)
#     clf = svm.SVC(kernel='linear', C=1)
#     clf = KNeighborsClassifier(n_neighbors=20)

    print("Fitting Data...")
    clf.fit(data, labels)
    
    print("Saving CLF...")
    pickle.dump(clf, open(pickle_directory + "clf_" + str(prm.params["chunk_size"].get()) + "_" + source +".p", "wb"))        
    
    print("Build Data Done")

