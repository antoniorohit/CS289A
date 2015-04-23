'''
Created on Apr 22, 2015

@author: antonio
'''
############# IMPORTS ############# 
from utils import *
from neuralNet import Digit_NN
import pylab as plt
import time

############# FILE STUFF ############# 
testFileMNIST = "./digit-dataset/test.mat"
trainFileMNIST = "./digit-dataset/train.mat"
    
trainMatrix = io.loadmat(trainFileMNIST)  # Dictionary
testMatrix = io.loadmat(testFileMNIST)  # Dictionary

############# GET DATA ############# 
print 20 * "#", "Getting Data", 20 * "#"
testData = np.array(testMatrix['test_images'])
testData = np.rollaxis(testData, 2, 0)  # move the index axis to be the first 

imageData = np.array(trainMatrix['train_images'])
imageData = np.rollaxis(imageData, 2, 0)  # move the index axis to be the first 

f = open("errors.txt", "rb")
indices = []
for line in f:
    line = line.strip()
    if line != "0" and line != "":
        indices.append(int(line.strip()))

print indices
        


for i in indices:
    print i
    plt.imshow(testData[i])
    plt.show()
    