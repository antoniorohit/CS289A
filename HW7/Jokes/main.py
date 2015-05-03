'''
Created on May 3, 2015

@author: antonio
'''
############# IMPORTS ############# 
import numpy as np
from scipy import io

############# FILE STUFF ############# 
trainFile = "./joke_data/joke_train.mat"
    
trainMatrix = io.loadmat(trainFile)  # Dictionary

############# GET DATA ############# 
print 20 * "#", "Getting Data", 20 * "#"
jokeData = np.array(trainMatrix['train'])

print np.shape(jokeData)