# Train a linear SVM using raw pixels as features. Plot the error rate on a validation
# set versus the number of training examples that you used to train your classier. The
# choices of the number of training examples should be 100, 200, 500, 1,000, 2,000, 5,000
# and 10,000. Make sure you set aside 10,000 other training points as a validation set.
# You should expect accuracies between 70% and 90% at this stage

############# IMPORTS ############# 

import scipy as sp
import numpy as np
from sklearn import svm
from scipy import io
import random
from skimage.io._plugins.qt_plugin import ImageLabel

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
C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]                    # array of values for parameter C
training_Size = [100, 200, 500, 1000, 2000, 5000, 10000]
for C_Value in C:
    print 50*'-'
    print "C Value:", C_Value
    print 50*'-'
    for elem in training_Size:
        if DEBUG:
            print 50*'-'
            print "Shuffled Data and Label shape: ", len(shuffledData), len(shuffledLabels)
        
        clf = svm.SVC(kernel='linear', C=C_Value)
        clf.fit(shuffledData[:elem], np.array(shuffledLabels[:elem]))
        
        predicted_Digits = clf.predict(shuffledData[50000:])
        actual_Digits = shuffledLabels[50000:]
        accuracy = 0.0
        for elem1, elem2 in zip(predicted_Digits, actual_Digits):
            if elem1 == elem2:
                accuracy+=1
        
        print "Training Size:", elem 
        print "Accuracy: ", 100.0*accuracy/len(predicted_Digits), "%"

############# PLOT RESULTS ############# 

    
print 50*'-'
print "End of File"