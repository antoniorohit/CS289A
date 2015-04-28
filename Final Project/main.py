'''
Created on Apr 13, 2015

High-Level Code which is called to test or train the system

@author: antonio
'''
from train import train
from test import test
from utils import cleanPickle
import parameters as prm
import os
import numpy as np
    

for chunkSize in prm.params["chunk_size"].all():
    print 60 * "-"        
    prm.params["chunk_size"].set(chunkSize)
    print 60 * "-"    
    
#     cleanPickle()
    
    print 20 * "#", "Training", 20 * "#"
    train()
    
    print 20 * "#", "Testing", 20 * "#"
    accuracy = test()
    
    print "Accuracy:", np.around(accuracy, 2), "%"
    

print 20 * "#", "End of Main. Thank you for Playing", 20 * "#"
