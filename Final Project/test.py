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

def test():  # only tests test_protocol data now    
    pickle_directory = prm.params["pickle_directory"].get()
    data_source= prm.params["data_source"].get()
    
    print "Loading CLF...."
    clf = pickle.load(open(pickle_directory + "clf_" + str(prm.params["chunk_size"].get()) + "_" + data_source + ".p", "rb"))
    print "Done loading CLF...."
    data, labels, rawData = getData_TestProtocol(source="test")
    
    predictedLabel = []
    for chunk in data:
        predictedGender = clf.predict(chunk)
        predictedLabel.append(predictedGender)
                
    accuracy = 0
    index = 0
    for (elem1, elem2) in zip(predictedLabel, labels):
        if elem1 == elem2:
            accuracy += 1
        else:
            if prm.params["device"].get() == "mac":
                write_wave((2, 2, 44100), np.int16(rawData[index]), './Errors/cleaned_audio_' + str(index) + '.wav')
            else:
                write_wave((1, 2, 44100), np.int16(rawData[index]), './Errors/cleaned_audio_' + str(index) + '.wav')
                
        index+=1

#     print "Predicted:", len(predictedLabel), sum(predictedLabel)
    
    return 100.0 * accuracy / len(predictedLabel)
