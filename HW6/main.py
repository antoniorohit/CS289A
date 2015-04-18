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

# imageData, imageLabels = getDataPickle(imageData, imageLabels)
# print imageLabels[0:11000]
imageData, imageLabels, _ = getDataNonMalik(zip(imageData, imageLabels))

# # shufTestData, _, _ = getDataMalik(0, testData, np.ones(len(testData)))
# shufTestData, _, _ = getDataNonMalik(zip(testData, [np.ones(len(testData))]))

dataShape = np.shape(imageData)
print dataShape

############# DATA PARTIONING ############# 
crossValidation_Data= []
crossValidation_Labels = []
k = 10 
lengthData = 5000
stepLength = k
for index in range(0,k):
    crossValidation_Data.append(imageData[index:lengthData:stepLength])
    crossValidation_Labels.append(imageLabels[index:lengthData:stepLength])

clf = Digit_NN(dataShape[1], n_hidden=200)

score = computeCV_Score(clf, crossValidation_Data, crossValidation_Labels, k)


print "Neural Net Score:", score, "%"

############# FOR KAGGLE ############# 
indices = np.array(range(1, len(shufTestData) + 1))
pred_labels = []
for elem in shufTestData:
    predictedLabel = clf.predict(np.matrix(elem))
    print np.around(predictedLabel)
    actual_label = np.nonzero(predictedLabel)
    print actual_label
    pred_labels.append(actual_label)

kaggle_format = np.vstack(((indices), pred_labels)).T
np.savetxt("./Results/spam.csv", kaggle_format, delimiter=",", fmt='%d,%d', header='Id,Category', comments='') 

