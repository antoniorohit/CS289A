'''
Created on Apr 16, 2015

Homework 6 (Neural Nets)

@author: antonio
'''
############# IMPORTS ############# 
from utils import *
from neuralNet import Digit_NN
import pylab as plt
import time

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
print 20*"#", "Getting Data", 20*"#"
testData = np.array(testMatrix['test_images'])
testData = np.rollaxis(testData, 2, 0)                # move the index axis to be the first 
testData_flat = []
for elem in testData:
    testData_flat.append(elem.flatten())
imageData = np.array(trainMatrix['train_images'])
imageData = np.rollaxis(imageData, 2, 0)                # move the index axis to be the first 
imageLabels = np.array(trainMatrix['train_labels'])

method = "malik"            # or "raw"

if method == "raw":
    imageData, imageLabels, _ = getDataNonMalik(zip(imageData, imageLabels))
    shufTestData, _, _ = getDataNonMalik(zip(testData, [np.ones(len(testData))]))
else:
    imageData, imageLabels = getDataPickle(imageData, imageLabels, "train")
    shufTestData, _, _ = getDataPickle(testData, np.ones(len(testData)), "test")

dataShape = np.shape(imageData)
print "Image Data Shape", dataShape

############# DATA PARTIONING ############# 
print 20*"#", "Cross Validation", 20*"#"
crossValidation_Data= []
crossValidation_Labels = []
k = 10 
lengthData = 6000
stepLength = k
for index in range(0,k):
    crossValidation_Data.append(imageData[index:lengthData:stepLength])
    crossValidation_Labels.append(imageLabels[index:lengthData:stepLength])

try:
    clf = pickle.load(open("./Results/nn_clf_" + method + ".p", "rb"))
except:
    clf = Digit_NN(dataShape[1], n_hidden=200)

score, clf = computeCV_Score(clf, crossValidation_Data, crossValidation_Labels, k)

print "Neural Net Score:", np.around(np.mean(score),1), "%"

print 20*"#", "Training on all Data...", 20*"#"
start = time.time()
clf.fit(imageData, imageLabels)
print "Training Time for entire NN:", time.time()-start

print "Saving the NN CLF..."
# pickle.dump(clf, open("./Results/" +"nn_clf_" + method + ".p", "wb"))        
print "Done saving the NN CLF!"

############# FOR KAGGLE ############# 
print 20*"#", "Predicting for Kaggle", 20*"#"
indices = np.array(range(1, len(shufTestData) + 1))
pred_labels = []
print np.shape(shufTestData)
for elem in shufTestData:
    pred_labels.append(clf.predict(np.matrix(elem)))

kaggle_format = np.vstack(((indices), pred_labels)).T
np.savetxt("./Results/digits.csv", kaggle_format, delimiter=",", fmt='%d,%d', header='Id,Category', comments='') 

print 20*"#", "The End !", 20*"#"
