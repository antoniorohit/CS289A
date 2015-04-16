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

prm.params["chunk_size"].set(.25)

# data_dir = "./Data/_r2h-20100822-kmh/wav/b028.wav"
data_dir = "./Data/aileen-20080831-dfq/wav/a0395.wav"
pickle_directory = prm.params["pickle_directory"].get()

clf = pickle.load(open(pickle_directory+"clf.p", "rb"))

params, rawSignal = get_RawSignal(data_dir)

rawSignal_framed = (frame_chunks(rawSignal))

predictedLabel = []
for chunk in rawSignal_framed:
    features = np.array(extractFeatures(chunk)).flatten()
    predictedLabel.append(clf.predict(features))
    
print len(predictedLabel), sum(predictedLabel)