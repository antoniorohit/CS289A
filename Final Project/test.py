import sys

sys.path.append("./feature_extraction")

import os
import parameters as prm
from utils import *
import cPickle as pickle

data_directory = prm.params["data_directory"].get()
prm.params["chunk_size"].set(1)
            
try:
    print "Loading the Data..."
    data = pickle.load(open(data_directory+"data.p", "rb"))
    labels = pickle.load(open(data_directory+"labels.p", "rb"))
    print "Data Loaded, Good to Go"
except Exception as excp:
    print excp
    data, labels = extract_Data(data_directory)
    # Save the data
    pickle.dump(data, open(data_directory+"data.p", "wb"))        
    pickle.dump(labels, open(data_directory+"labels.p", "wb"))        

print sum(labels), len(labels)

print 20*"*", "The End", 20*"*"

