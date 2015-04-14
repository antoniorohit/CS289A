import sys

sys.path.append("./feature_extraction")

import os
import parameters as prm
from utils import *
import cPickle as pickle

data_directory = prm.params["data_directory"].get()
prm.params["chunk_size"].set(1)
            
try:
    print "Loading the Data... (may take a while)"
    data = pickle.load(open(data_directory+"data.p", "rb"))
    labels = pickle.load(open(data_directory+"labels.p", "rb"))
    print "Data Loaded, Good to Go!"
except Exception as excp:
    print "Exception:", excp, Exception
    data, labels = extract_Data(data_directory)
    print "Saving the Data... (may take a while)"
    pickle.dump(data, open(data_directory+"data.p", "wb"))        
    pickle.dump(labels, open(data_directory+"labels.p", "wb"))        
    print "Data Saved, Good to Go!"

print labels
print "Shapes of Data and Labels", np.shape(data), np.shape(labels)
print sum(labels)

print 20*"*", "The End", 20*"*"

