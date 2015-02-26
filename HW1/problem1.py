# Train a linear SVM using raw pixels as features. Plot the error rate on a validation
# set versus the number of training examples that you used to train your classier. The
# choices of the number of training examples should be 100, 200, 500, 1,000, 2,000, 5,000
# and 10,000. Make sure you set aside 10,000 other training points as a validation set.
# You should expect accuracies between 70% and 90% at this stage
from dynd._pydynd import linspace


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
from scipy import signal
import numpy as np
from sklearn import svm
from scipy import io
import random
from skimage.io._plugins.qt_plugin import ImageLabel
from sklearn.metrics import confusion_matrix
import pylab as plt
from sklearn import cross_validation

MALIK = True
DEBUG = False
############# FILE STUFF ############# 
testFileMNIST = "./digit-dataset/test.mat"
trainFileMNIST = "./digit-dataset/train.mat"


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

############# 
# Ink Normalization
############# 
if MALIK == True:
    i = 0
    for element in imageData:
        if np.linalg.norm(imageData[i]) != 0:
            imageData[i] = (imageData[i]/np.linalg.norm(imageData[i]))
        i+=1
else:
    pass

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

############# 
# ORIENTATION HISTOGRAM
############# 
shuffledData = []
shuffledLabels = []
bins = np.linspace(-np.pi, np.pi, 12)

if MALIK == True:
    for elem in imageComplete:
        gradx, grady = np.gradient(elem[0])
        mag = np.sqrt(np.square(gradx) + np.square(grady))
        cell = (1.0)*np.ones((4,4))
        ori = (np.arctan2(grady, gradx))
        # Aggregate over a cell 4x4 etc moved by half cell size
#         ori = signal.convolve2d(ori, cell, 'same')
#         ori = np.histogram(ori.flatten(), bins)
        
#         print ori[0]
        
#         ori = ori[0]/np.linalg.norm(ori[0])
        if np.linalg.norm(mag) != 0:
            mag = (mag-np.mean(mag))/np.linalg.norm(mag)
        
        ori = (ori-np.mean(ori))/np.linalg.norm(ori)
        
        shuffledData.append(np.append(mag.flatten(), ori.flatten()))
        shuffledLabels.append((elem[1][0]))
    
else:
    for elem in imageComplete:
        shuffledData.append((elem[0]).flatten())                # Use a simple array of pixels as the feature
        shuffledLabels.append((elem[1][0]))
    
# NOTE: Set aside 50,000-60,000 to validate

# Plot the distribution of digits in Validation Data
plt.title('Histogram for Validation Data')
plt.ylabel('Count')
plt.xlabel('Digit Label')
plt.hist(shuffledLabels[50000:])
plt.savefig("./Results/ValidationData_Hist.png")

############# TRAIN SVM ############# 
print 50*'='
print "SVM TRAINING"
print 50*'='

errorRate_array = []
C = [0.75, 1, 1.5, 1.75, 2, 2.25, 2.5]                    # array of values for parameter C
training_Size = [100, 200, 500, 1000, 2000, 5000, 10000]
for elem in training_Size:
    if DEBUG:
        print 50*'-'
        print "Shuffled Data and Label shape: ", len(shuffledData), len(shuffledLabels)
    
    clf = svm.SVC(kernel='linear', C=C[4])
    clf.fit(shuffledData[:elem], np.array(shuffledLabels[:elem]))
    
    predicted_Digits = clf.predict(shuffledData[50000:])
    actual_Digits = shuffledLabels[50000:]
    accuracy = 0.0
    for elem1, elem2 in zip(predicted_Digits, actual_Digits):
        if elem1 == elem2:
            accuracy+=1
    
    errorRate_array.append(100-100.0*accuracy/len(predicted_Digits))
    print "Training Size:", elem 
    print "Error Rate: ", errorRate_array[-1], "%"
    print 50*'-'

    cm = confusion_matrix(actual_Digits, predicted_Digits)
############# PLOT RESULTS #############     
    # Show confusion matrix in a separate window
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix')
    fig.colorbar(cax)
    plt.ylabel('True Digit')
    plt.xlabel('Predicted Digit')
    plt.savefig("./Results/" + str(elem)+"_CM.png")

# Remove color bar
fig.delaxes(fig.axes[0]) 

# Plot error rate vs training size
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Error Rate Vs Training Size')
ax.set_ylabel('Training Size')
ax.set_xlabel('Error Rate')
ax.plot(training_Size, errorRate_array)
plt.savefig("./Results/ErrorRate_TrainingSize.png")
####################################### 

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
lengthData = 10000
stepLength = k
for index in range(0,k):
    crossValidation_Data.append(shuffledData[index:lengthData:stepLength])
    crossValidation_Labels.append(shuffledLabels[index:lengthData:stepLength])

if DEBUG:
    print "Lengths of CV Data and Labels: ", np.array(crossValidation_Data).shape, np.array(crossValidation_Labels).shape
    print 50*'-'

scoreBuffer = []

############# CROSS-VALIDATION ############# 
for C_Value in C:
    clf = svm.SVC(kernel='linear', C=C_Value)
    scores = computeCV_Score(clf, crossValidation_Data, crossValidation_Labels, k)
    scoreBuffer.append((scores).mean())
    if DEBUG:
        print "C Value:", C_Value, "Accuracy: %0.2f (+/- %0.2f)" % ((scores).mean(), np.array(scores).std() / 2)
        print 50*'-'

maxScore = np.max(np.array(scoreBuffer))
maxScore_Index = scoreBuffer.index(maxScore)

# Train SVM using best C value
clf = svm.SVC(kernel='linear', C=C[maxScore_Index])
clf.fit(shuffledData[:10000], np.array(shuffledLabels[:10000]))
# Predict digits
predicted_Digits = clf.predict(shuffledData[50000:])
actual_Digits = shuffledLabels[50000:]
# Compute Accuracy
accuracy = 0.0
for elem1, elem2 in zip(predicted_Digits, actual_Digits):
    if elem1 == elem2:
        accuracy+=1

print "Using Custom CV Function"
print "Best C Value:", C[maxScore_Index], "Accuracy for that C:", (100.0*accuracy/len(predicted_Digits))
print 50*'-'

# FOR KAGGLE
indices = np.array(range(1,len(testData_flat)+1))
kaggle_format =  np.vstack(((indices), (clf.predict(testData_flat)))).T

if DEBUG:
    print indices.shape, clf.predict(testData_flat).shape
    print kaggle_format.shape, kaggle_format

np.savetxt("./Results/Digits.csv", kaggle_format, delimiter=",", fmt = '%d,%d',   header = 'Id,Category', comments='') 

############# USING BUILT IN FUNCTION ############# 
# for C_Value in C:
#     clf = svm.SVC(kernel='linear', C=C_Value)
#     scores = cross_validation.cross_val_score(clf, shuffledData[:10000], shuffledLabels[:10000], cv=10)
#     if DEBUG:
#         print "C Value:", C_Value, "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)
#         print 50*'-'
# 
# maxScore = scoreBuffer.max()
# maxScore_Index = scoreBuffer.index(maxScore)
# print "Using BuiltIn CV Function"
# print "Best C Value:", C[maxScore_Index], "Accuracy for that C:", maxScore
# print 50*'-'


print 50*'='
print "End of File"
print 50*'='
