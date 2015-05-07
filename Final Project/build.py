'''
Created on May 7, 2015

@author: antonio
'''
import os
import parameters as prm
from utils import *
import cPickle as pickle
from sklearn import svm, tree, ensemble
import random


def build(source="test_protocol"):
    pickle_directory = prm.params["pickle_directory"].get()
    print("Building CLF...")
    if source == "test_protocol":
        data, labels, rawData = getTrainData_Pickle("test_protocol")
    else:
        data, labels, rawData = getTrainData_Pickle("voxforge")
    
    clf = ensemble.RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=10)
#     clf = svm.SVC(kernel=kernel, C=C[maxScore_Index])
#     clf = KNeighborsClassifier(n_neighbors=num_neighbors[maxScore_Index])

    clf.fit(data, labels)
    pickle.dump(clf, open(pickle_directory + "clf_" + str(prm.params["chunk_size"].get()) + "_" + source +".p", "wb"))        
    print("Build Data Done")

