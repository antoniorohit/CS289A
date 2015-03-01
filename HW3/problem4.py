# Train a linear SVM using raw pixels as features. Plot the error rate on a validation
# set versus the number of training examples that you used to train your classier. The
# choices of the number of training examples should be 100, 200, 500, 1,000, 2,000, 5,000
# and 10,000. Make sure you set aside 10,000 other training points as a validation set.
# You should expect accuracies between 70% and 90% at this stage
from dynd._pydynd import linspace
import matplotlib.cm as cm
from scipy.stats import norm

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
                    else:
                        print data[i].shape
                        plt.imshow(data[i])
                        plt.show()
                
            j+=1
        scores.append(100.0*accuracy/((folds-1)*len(predicted_Digits)))
        i+=1
    return np.array(scores)
###################################

############# IMPORTS ############# 

def train_gauss(data, labels):
    ############# 
    # Fit gaussian to each digit
    ############# 
    image = np.zeros((28,28))
    gauss_list = []
    overall_cov = np.zeros((784, 784))
    all_cov = []
    all_prior = []
    all_mu = []
    
    for i in range(0,10):
        data_label = []
        index  = 0
        for label in labels:
            if(label == i):
                data_label.append(data[index].flatten())
            index += 1
        # data label contains all the data of a certain number
        # transpose it so that we can take the average across 
        # all data
        data_label = np.transpose(np.array(data_label))
        
        # mean mu for label 
        mu = [np.mean(data_label[j]) for j in range(0, 28*28)]
        
        # sample covariance
        cov = np.zeros((784, 784))
        data_label = data_label.T 
        for elem in (data_label):
            cov += np.mat(elem-mu).T*np.mat(elem-mu)
        cov = cov/len(data_label)
    
    
        # prior probability 
        prior = len(data_label)/60000.0
        
        # visualize this mean
    #     for j in range(0, 28):
    #         image[j] = mu[28*j:28*j+28]
    
    #     plt.figure()
    #     plt.imshow(image)
    #     plt.figure()
    #     plt.imshow(cov)
    #     plt.show()
            
        print prior, i
        all_cov.append(cov)
        all_prior.append(prior)
        all_mu.append(mu)
    return all_mu, all_cov, all_prior

def gauss_predict(data, all_mu, overall_cov, all_prior):
    labelled_list = []
    for elem in data:
        prob_list = []
        for label in range(0, 10):
            mu = all_mu[label]
            cov = all_cov[label]
            n = norm(loc=mu, scale=cov)
            prob_list.append(np.sum(n.logpdf(elem))*all_prior[label])
        labelled_list.append(prob_list.index(max(prob_list)))
    return labelled_list

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
testData = np.array(testMatrix['test_image'])
testData = np.rollaxis(testData, 2, 0)                # move the index axis to be the first 
testData_flat = []
for elem in testData:
    testData_flat.append(elem.flatten())
imageData = np.array(trainMatrix['train_image'])
imageData = np.rollaxis(imageData, 2, 0)                # move the index axis to be the first 
imageLabels = np.array(trainMatrix['train_label'])


############# 
# Ink Normalization
############# 
i = 0
for element in imageData:
    if np.linalg.norm(imageData[i]) != 0:
        imageData[i] = (imageData[i]/np.linalg.norm(imageData[i]))
    i+=1

imageComplete = zip(imageData, imageLabels)

random.shuffle(imageComplete)
shuffledData = []
shuffledLabels = []
for elem in imageComplete:
    shuffledData.append((elem[0]).flatten())                # Use a simple array of pixels as the feature
    shuffledLabels.append((elem[1][0]))
    
errorRate_array = []
C = np.linspace(1,3,16)                   # array of values for parameter C
training_Size = [100, 200, 500, 1000, 2000,]# 5000, 10000, 15000]
for elem in training_Size:
    all_mu, all_cov, all_prior = train_gauss(shuffledData[:elem], np.array(shuffledLabels[:elem]))
    overall_cov = np.mean(all_cov)   
    
    predicted_Digits = gauss_predict(shuffledData[50000:], all_mu, overall_cov, all_prior)
    actual_Digits = shuffledLabels[50000:]
    accuracy = 0.0
    for elem1, elem2 in zip(predicted_Digits, actual_Digits):
        if elem1 == elem2:
            accuracy+=1
    
    errorRate_array.append(100-100.0*accuracy/len(predicted_Digits))
    print "Training Size:", elem 
    print "Error Rate: ", errorRate_array[-1], "%"
    print 50*'-'

############# PLOT RESULTS #############     

# Plot error rate vs training size
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Error Rate Vs Training Size')
ax.set_ylabel('Training Size')
ax.set_xlabel('Error Rate')
ax.plot(training_Size, errorRate_array)
plt.savefig("./Results/ErrorRate_TrainingSize.png")
####################################### 
        
        
print 5*"-The End-"