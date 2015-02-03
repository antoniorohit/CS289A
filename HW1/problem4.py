# Train a linear SVM using raw pixels as features. Plot the error rate on a validation
# set versus the number of training examples that you used to train your classier. The
# choices of the number of training examples should be 100, 200, 500, 1,000, 2,000, 5,000
# and 10,000. Make sure you set aside 10,000 other training points as a validation set.
# You should expect accuracies between 70% and 90% at this stage

###################################
# Function to calculate cross validation scores 
# Input: SVC object, data, labels, num folds
# Output: Array of scores averaged for each fold
###################################
def computeCV_Score(clf, data, labels, folds):
    i = 0
    j = 0
    accuracy = 0.0
    scores = []
    clf_local = clf
    # For each fold trained on...
    for i in range(folds):
        # Initialize variables
        clf_local = clf
        j = 0
        accuracy = 0

        clf_local.fit(data[i], labels[i])
        # For each validation performed (k-1 total) on a fold
        for j in range(folds):
            if(j!=i):
                predicted_Digits = clf_local.predict(data[j])
                for (elem1, elem2) in zip(predicted_Digits, labels[j]):
                    if elem1 == elem2:
                        accuracy+=1                
            j+=1
        scores.append(100.0*accuracy/((folds-1)*len(predicted_Digits)))
        i+=1
    return np.array(scores)
###################################

############# IMPORTS ############# 

import scipy as sp
import numpy as np
from sklearn import svm
from scipy import io
import random
from skimage.io._plugins.qt_plugin import ImageLabel
from sklearn.metrics import confusion_matrix
import pylab as plt
from sklearn import cross_validation

DEBUG = True
############# FILE STUFF ############# 
File_Spam = "./spam-dataset/spam_data.mat"


trainMatrix = io.loadmat(File_Spam)                 # Dictionary

if DEBUG:
    print 50*'-'
    print trainMatrix

############# GET DATA ############# 
trainingData = np.array(trainMatrix['training_data'])
trainingLabels = np.array(trainMatrix['training_labels'][0])
testData= np.array(trainMatrix['test_data'][0])
trainingComplete = zip(trainingData, trainingLabels)

if DEBUG:
    print 50*'-'
    print ("Shapes of data and labels: ", trainingData.shape, 
                                    trainingLabels.shape, len(trainingComplete))
        
C = [ 0.1, 1, 5, 10, 50]                    # array of values for parameter C

#########################################################
# CROSS VALIDATION 
#########################################################
print 50*'='
print "CROSS VALIDATION"
print 50*'='

############# DATA PARTIONING ############# 
crossValidation_Data= []
crossValidation_Labels = []
k = 10 
stepLength = k
for index in range(0,k):
    crossValidation_Data.append(trainingData[index:-1:stepLength])
    crossValidation_Labels.append(trainingLabels[index:-1:stepLength])

if DEBUG:
    print "Lengths of CV Data and Labels: ", np.array(crossValidation_Data).shape, np.array(crossValidation_Labels).shape
    print 50*'-'

scoreBuffer = []

############# CROSS-VALIDATION ############# 
for C_Value in C:
    clf = svm.SVC(kernel='linear', C=C_Value)
    scores = computeCV_Score(clf, crossValidation_Data, crossValidation_Labels, k)
    scoreBuffer.append((scores).mean)
    if DEBUG:
        print "C Value:", C_Value, "Accuracy: %0.2f (+/- %0.2f)" % ((scores).mean(), np.array(scores).std() / 2)
        print 50*'-'

maxScore = np.max(scoreBuffer)
maxScore_Index = scoreBuffer.index(maxScore)
print "Best C Value:", C[maxScore_Index], "Accuracy for that C:", maxScore
print 50*'-'

############# TRAIN SVM ############# 
print 50*'='
print "SVM TRAINING"
print 50*'='

if DEBUG:
    print 50*'-'
    print "Shuffled Data and Label shape: ", len(trainingData), len(trainingLabels)

clf = svm.SVC(kernel='linear', C=C[2])
clf.fit(trainingData, np.array(trainingLabels))

predicted_Digits = clf.predict(testData)
actual_Digits = trainingLabels
accuracy = 0.0
for elem1, elem2 in zip(predicted_Digits, actual_Digits):
    if elem1 == elem2:
        accuracy+=1

print "Accuracy: ", 100.0*accuracy/len(predicted_Digits), "%"
print 50*'-'

cm = confusion_matrix(actual_Digits, predicted_Digits)
############# PLOT RESULTS #############     
# Show confusion matrix in a separate window
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True Digit')
plt.xlabel('Predicted Digit')
plt.savefig("./ResultsSpam/" +"_CM.png")

####################################### 





print 50*'-'
print "End of File"
print 50*'-'
