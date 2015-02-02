# Train a linear SVM using raw pixels as features. Plot the error rate on a validation
# set versus the number of training examples that you used to train your classier. The
# choices of the number of training examples should be 100, 200, 500, 1,000, 2,000, 5,000
# and 10,000. Make sure you set aside 10,000 other training points as a validation set.
# You should expect accuracies between 70% and 90% at this stage

# Function to calculate cross validation scores 
# Input: SVC object, data, labels, num folds
# Output: Array of scores averaged for each fold
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
    return scores


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

DEBUG = False
############# FILE STUFF ############# 
testFileMNIST = "./digit-dataset/test.mat"
trainFileMNIST = "./digit-dataset/train.mat"

trainMatrix = sp.io.loadmat(trainFileMNIST)                 # Dictionary
testMatrix = sp.io.loadmat(testFileMNIST)                   # Dictionary

if DEBUG:
    print 50*'-'
    print trainMatrix, testMatrix

############# GET DATA ############# 
imageData = np.array(trainMatrix['train_images'])
imageData = np.rollaxis(imageData, 2, 0)                # move the index axis to be the first 
imageLabels = np.array(trainMatrix['train_labels'])

imageComplete = zip(imageData, imageLabels)

if DEBUG:
    print 50*'-'
    print ("Shapes of image data and labels: ", imageData.shape, 
                                    imageLabels.shape, len(imageComplete))
    
    print "Image/Digit 10000:\n", imageComplete[20000]
    
############# SET ASIDE VALIDATION DATA (10,000) ############# 
# VERIFY ORDER
if DEBUG:    
    i = 0
    print 50*'-'
    for elem in imageComplete:
        print elem[1]
        i+=1
        if i > 5000:
            break
        
# SHUFFLE THE IMAGES
random.shuffle(imageComplete)

# VERIFY SHUFFLE
if DEBUG:
    i = 0    
    print 50*'-'
    for elem in imageComplete:
        print elem[1]
        i+=1
        if i > 5000:
            break

shuffledData = []
shuffledLabels = []

for elem in imageComplete:
    shuffledData.append((elem[0]).flatten())                # Use a simple array of pixels as the feature
    shuffledLabels.append((elem[1][0]))

# NOTE: Set aside 50,000-60,000 to validate

############# TRAIN SVM ############# 
print 50*'-'
print "SVM TRAINING"
print 50*'-'

C = [0.000000001, 0.000001, 0.001, 1, 1000]                    # array of values for parameter C
training_Size = [100, 200, 500, 1000]#, 2000, 5000, 10000]
for elem in training_Size:
    if DEBUG:
        print 50*'-'
        print "Shuffled Data and Label shape: ", len(shuffledData), len(shuffledLabels)
    
    clf = svm.SVC(kernel='linear', C=C[3])
    clf.fit(shuffledData[:elem], np.array(shuffledLabels[:elem]))
    
    predicted_Digits = clf.predict(shuffledData[50000:])
    actual_Digits = shuffledLabels[50000:]
    accuracy = 0.0
    for elem1, elem2 in zip(predicted_Digits, actual_Digits):
        if elem1 == elem2:
            accuracy+=1
    
    print "Training Size:", elem 
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
    plt.savefig("./Results/" + str(elem)+"_CM.png")

####################################### 

#########################################################
# CROSS VALIDATION 
#########################################################
print 50*'-'
print "CROSS VALIDATION"
print 50*'-'

############# DATA PARTIONING ############# 
crossValidation_Data= []
crossValidation_Labels = []
k = 10 
lengthData = 10000
stepLength = k
for index in range(0,k):
    crossValidation_Data.append(shuffledData[index:lengthData:stepLength])
    crossValidation_Labels.append(shuffledLabels[index:lengthData:stepLength])

if DEBUG:
    print "Lengths of CV Data and Labels: ", np.array(crossValidation_Data).shape, np.array(crossValidation_Labels).shape
    print 50*'-'

############# CROSS-VALIDATION ############# 
for C_Value in C:
    clf = svm.SVC(kernel='linear', C=C_Value)
    scores = computeCV_Score(clf, crossValidation_Data, crossValidation_Labels, k)
    print "C Value:", C_Value, "Accuracy: %0.2f (+/- %0.2f)" % (np.array(scores).mean(), np.array(scores).std() / 2)
    print 50*'-'

############# USING BUILT IN FUNCTION ############# 
for C_Value in C:
    clf = svm.SVC(kernel='linear', C=C_Value)
    scores = cross_validation.cross_val_score(clf, shuffledData[:10000], shuffledLabels[:10000], cv=10)
    print "C Value:", C_Value, "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
    print 50*'-'


print 50*'-'
print "End of File"
print 50*'-'
