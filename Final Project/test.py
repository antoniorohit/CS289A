'''
Created on Apr 13, 2015

Implements the ML - training the model

@author: antonio
'''
import sys
sys.path.append("./feature_extraction")

from utils import *
import cPickle as pickle
from extract_features import extractFeatures
import os 

def test():  # primarily tests testprotocol data now
#     male_file = "./Data/_r2h-20100822-kmh/wav/b0284.wav"  # Male
#     female_file = "./Data/aileen-20080831-dfq/wav/a0395.wav"  # Female
    
    pickle_directory = prm.params["pickle_directory"].get()
    
#     print "Loading CLF...."
    clf = pickle.load(open(pickle_directory + "clf.p", "rb"))
#     print "Done loading CLF...."
    data, labels, rawData = getData_TestProtocol(source="test")
    
    predictedLabel = []
    for chunk in data:
        predictedGender = clf.predict(chunk)
        predictedLabel.append(predictedGender)
        
#         if sum(predictedGender) * 1. / len(predictedGender) < 0.5:
#             print "Female", sum(predictedLabel) * 1. / len(predictedLabel)
#         else:
#             print "Male", sum(predictedLabel) * 1. / len(predictedLabel)
        
    accuracy = 0
    for (elem1, elem2) in zip(predictedLabel, labels):
        if elem1 == elem2:
            accuracy += 1
        else:
            pass

    print "Predicted:", len(predictedLabel), sum(predictedLabel)
    
    return 100.0 * accuracy / len(predictedLabel)
