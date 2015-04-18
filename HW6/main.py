'''
Created on Apr 16, 2015

Homework 6 (Neural Nets)

@author: antonio
'''
############# IMPORTS ############# 
from utils import *
from neuralNet import Digit_NN
import pylab as plt

DEBUG = False
############# FILE STUFF ############# 
testFileMNIST = "./digit-dataset/test.mat"
trainFileMNIST = "./digit-dataset/train.mat"

def my_kernel(x, y, first_dim):
    
    hist = np.sum(np.absolute(np.minimum(x[first_dim:],y[first_dim:])))
    lin = np.dot(x[:first_dim], y[:first_dim].T)
    return hist + lin
    
trainMatrix = io.loadmat(trainFileMNIST)                 # Dictionary
testMatrix = io.loadmat(testFileMNIST)                   # Dictionary

if DEBUG:
    print 50*'-'
    print trainMatrix, testMatrix

############# GET DATA ############# 
testData = np.array(testMatrix['test_images'])
testData = np.rollaxis(testData, 2, 0)                # move the index axis to be the first 
testData_flat = []
for elem in testData:
    testData_flat.append(elem.flatten())
imageData = np.array(trainMatrix['train_images'])
imageData = np.rollaxis(imageData, 2, 0)                # move the index axis to be the first 
imageLabels = np.array(trainMatrix['train_labels'])

# shuffledData, shuffledLabels = getDataPickle(imageData, imageLabels)
shuffledData, shuffledLabels, _ = getDataNonMalik(zip(imageData, imageLabels))

dataShape = np.shape(shuffledData)
print dataShape

############# DATA PARTIONING ############# 
crossValidation_Data= []
crossValidation_Labels = []
k = 10 
lengthData = 1000
stepLength = k
for index in range(0,k):
    crossValidation_Data.append(shuffledData[index:lengthData:stepLength])
    crossValidation_Labels.append(shuffledLabels[index:lengthData:stepLength])

clf = Digit_NN(dataShape[1], n_hidden=200)
score = computeCV_Score(clf, crossValidation_Data, crossValidation_Labels, k)


print "Neural Net Score:", score, "%"


