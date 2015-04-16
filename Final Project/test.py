'''
Created on Apr 13, 2015

Implements the ML - training the model

@author: antonio
'''
import sys
sys.path.append("./feature_extraction")

from utils import *
import cPickle as pickle
from sklearn import svm, tree, ensemble 
from extract_features import extractFeatures
import os 

prm.params["chunk_size"].set(1)

male_file = "./Data/_r2h-20100822-kmh/wav/b0284.wav"                # Male
female_file = "./Data/aileen-20080831-dfq/wav/a0395.wav"            # Female

pickle_directory = prm.params["pickle_directory"].get()

print "Loading CLF...."
clf = pickle.load(open(pickle_directory+"clf.p", "rb"))
print "Done loading CLF...."


for filename in os.listdir("./TestData"):
    print filename
    params, rawSignal = get_RawSignal("./TestData/"+filename)
    
    print params
    
    prm.params["sample_rate"].set(params[2])

    rawSignal_framed = (frame_chunks(rawSignal))
    
    predictedLabel = []
    for chunk in rawSignal_framed:
        features = np.array(extractFeatures(chunk)).flatten()
        predictedLabel.append(clf.predict(features))
        
    if sum(predictedLabel)*1./len(predictedLabel) < 0.95:
        print "Female", sum(predictedLabel)*1./len(predictedLabel)
    else:
        print "Male", sum(predictedLabel)*1./len(predictedLabel)