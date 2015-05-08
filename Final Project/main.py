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
from build import build

source = "voxforge"

results = []

for device in prm.params["device"].all():
    print 60 * "*"        
    prm.params["device"].set(device)
    print 60 * "*"    
    for chunkSize in prm.params["chunk_size"].all():
        print 60 * "*"        
        prm.params["chunk_size"].set(chunkSize)
        print 60 * "*"    
        
        cleanPickle()       # cleans pickle and error directory (except for vf)
        
        print 20 * "#", "Building", 20 * "#"
        build(source)
        
#         print 20 * "#", "Training", 20 * "#"
#         train(source)       # test_protocol or voxforge
        
        print 20 * "#", "Testing", 20 * "#"
        accuracy = np.around(test(source), 2)
        
        print "Accuracy:", (accuracy), "%"
        
        results.append([device, chunkSize, np.around(accuracy,2)])
        
print results

print 20 * "*", "End of Main. Thank you for Playing", 20 * "*"

