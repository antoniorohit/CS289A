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

############# FILE STUFF ############# 
testFileMNIST = "./digit-dataset/test.mat"
trainFileMNIST = "./digit-dataset/train.mat"
    
trainMatrix = io.loadmat(trainFileMNIST)                 # Dictionary
testMatrix = io.loadmat(testFileMNIST)                   # Dictionary

############# GET DATA ############# 
print 20*"#", "Getting Data", 20*"#"
testData = np.array(testMatrix['test_images'])
testData = np.rollaxis(testData, 2, 0)                # move the index axis to be the first 

imageData = np.array(trainMatrix['train_images'])
imageData = np.rollaxis(imageData, 2, 0)                # move the index axis to be the first 
imageLabels = np.array(trainMatrix['train_labels'])

############# PROCESS DATA ############# 
features = "raw"            # or "raw"

if features == "raw":
    # Non malik - raw, shuffled, and labels converted to one-of-10-high format
    imageData, imageLabels, _ = getDataNonMalik(zip(imageData, imageLabels))
    shufTestData, _, _ = getDataNonMalik(zip(testData, np.ones((len(testData),1))))
else:
    # Malik - HOG, shuffled, and labels converted to one-of-10-high format
    imageData, imageLabels = getDataPickle(imageData, imageLabels, "train")
    shufTestData, _ = getDataPickle(testData, np.ones(len(testData)), "test")

# Shape of data determines number of input layers
dataShape = np.shape(imageData)
print "Image Data Shape", dataShape
print "Test Data Shape", np.shape(shufTestData)

############# DATA PARTIONING ############# 
print 20*"#", "Cross Validation", 20*"#"
crossValidation_Data= []
crossValidation_Labels = []
k = 10 
lengthData = 20000          # do CV with a subset of the entire dataset so it takes less time (eg 5-10k samples)
stepLength = k
for index in range(0,k):
    crossValidation_Data.append(imageData[index:lengthData:stepLength])
    crossValidation_Labels.append(imageLabels[index:lengthData:stepLength])
if lengthData < 50000:
    validationSet_Data = imageData[50000:]
    validationSet_Labels = imageLabels[50000:]

print "CV Data Shape:", np.shape(crossValidation_Data)

# try:    # try loading the old clf if it exists
#     clf = pickle.load(open("./Results/nn_clf_" + features + ".p", "rb"))
# except: # clf wasn't found - initialize new NN with ninput neurons and 200 neurons in hidden layer
clf = Digit_NN(dataShape[1], n_hidden=200, cost="MSE")

print "Gamma:", clf.gamma

score, clf = computeCV_Score(clf, crossValidation_Data, crossValidation_Labels, k)

print "NN Cross-Val Score:", np.around(np.mean(score),1), "%"

############# VALIDATION SET ############# 
print 20*"#", "Validation Set Accuracy", 20*"#"
pred_labels = clf.predict(validationSet_Data)
accuracy = 0
for (elem1, elem2) in zip(pred_labels, validationSet_Labels):
    elem2 = elem2.tolist().index(1)
    if elem1 == elem2:
        accuracy+=1
    else:
        pass

print "Validation Set Accuracy:", 100.0*accuracy/len(pred_labels), "%"

############# MEGA TRAINING ############# 
print 20*"#", "Training on all Data...", 20*"#"
start = time.time()
clf.fit(imageData, imageLabels)
print "Training Time for entire NN:", np.around((time.time()-start)/60., 1), "minutes"      # 13 minutes

try:    # @TODO can the CLF really be saved?
    print "Saving the NN CLF..."
    pickle.dump(clf, open("./Results/" +"nn_clf_" + features + ".p", "wb"))        
    print "Done saving the NN CLF!"
except Exception, e:
    print str(e)
    
############# VALIDATION SET (AGAIN) ############# 
print 20*"#", "Validation Set Accuracy", 20*"#"
pred_labels = clf.predict(validationSet_Data)
accuracy = 0
for (elem1, elem2) in zip(pred_labels, validationSet_Labels):
    elem2 = elem2.tolist().index(1)
    if elem1 == elem2:
        accuracy+=1
    else:
        pass

print "Validation Set Accuracy:", 100.0*accuracy/len(pred_labels), "%"

############# FOR KAGGLE ############# 
print 20*"#", "Predicting for Kaggle", 20*"#"
indices = np.array(range(1, len(shufTestData) + 1))
print np.shape(shufTestData)
pred_labels = clf.predict(shufTestData)

kaggle_format = np.vstack(((indices), pred_labels)).T
np.savetxt("./Results/digits.csv", kaggle_format, delimiter=",", fmt='%d,%d', header='Id,Category', comments='') 

print 20*"#", "The End !", 20*"#"
