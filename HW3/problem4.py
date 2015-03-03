
from scipy.stats import multivariate_normal as norm

###################################
# Function to calculate cross validation scores 
# Input: SVC object, data, labels, num folds
# Output: Array of scores averaged for each fold
###################################
def computeCV_Score(gauss_params, data, labels, folds, weight):
    i = 0
    j = 0
    accuracy = 0.0
    scores = []
    (all_mu, all_cov, all_prior) = gauss_params
    # For each fold trained on...
    for i in range(folds):
        # Initialize variables
        j = 0
        accuracy = 0
        for j in range(folds):
            if(j!=i):
                predicted_Digits = gauss_predict(data[j], all_mu, all_cov, all_prior, False, weight)
                for (elem1, elem2) in zip(predicted_Digits, labels[j]):
                    if elem1 == elem2:
                        accuracy+=1
                    else:
                        pass
#                         print data[i].shape
#                         plt.imshow(data[i])
#                         plt.show()
                
            j+=1
        scores.append(100.0*accuracy/((folds-1)*len(predicted_Digits)))
        i+=1
    return np.array(scores)
###################################

############# IMPORTS ############# 

def train_gauss(im_data, labels):
    ############# 
    # Fit gaussian to each digit
    ############# 
    all_cov = []
    all_prior = []
    all_mu = []
    
    for i in range(0,10):
        data_label = []
        index  = 0
        for label in labels:
            if(label == i):
                data_label.append(im_data[index].flatten())
            index += 1
        # data label contains all the data of a certain number
        # transpose it so that we can take the average across 
        # all data
        data_label = np.transpose(np.array(data_label))
        
        # mean mu for label 
        mu = [np.mean(data_label[j]) for j in range(0,len(data_label))]
        
        # sample covariance
        cov = np.zeros((len(data_label), len(data_label)))
        data_label = data_label.T 
        for elem in (data_label):
            cov += np.mat(elem-mu).T*np.mat(elem-mu)
        cov = cov/len(data_label)
    
    
        # prior probability 
        prior = float(len(data_label))/(len(im_data))
        
############# 
# Part c of question
############# 
        # visualize this mean and covariance
#         image = np.zeros((28,28))
#         for j in range(0, 28):
#             image[j] = mu[28*j:28*j+28]
#       
#         plt.figure()
#         plt.imshow(image)
#         plt.figure()
#         plt.imshow(cov)
#         plt.show()
            

############# 
# Part bCov_Co of question
############# 
        print np.around(prior, 3), i
        all_cov.append(cov)
        all_prior.append(prior)
        all_mu.append(mu)
    return all_mu, all_cov, all_prior

def gauss_predict(data, all_mu, all_cov, all_prior, overall = True, weight = 0.001):
    labelled_list = []
    
    # Create PDFs
    n = []
    for label in range(0, 10):
        mu = all_mu[label]
        if overall == True:
            cov = np.mean(all_cov)
        else:
            # add small value to diag to remove singularity
            small_value = weight*np.eye(len(all_cov[0]))
            cov = all_cov[label]+small_value
        n.append(norm(mean=mu, cov=(cov)))
                
    for elem in data:
        prob_list = []
        for label in range(0, 10):
            prob_list.append(np.sum(n[label].logpdf(elem))*all_prior[label])
        labelled_list.append(prob_list.index(max(prob_list)))
    
    return labelled_list

import numpy as np
from scipy import io
import random
import pylab as plt

DEBUG = False
############# FILE STUFF ############# 
testFileMNIST = "./digit-dataset/kaggle.mat"
trainFileMNIST = "./digit-dataset/train.mat"


trainMatrix = io.loadmat(trainFileMNIST)                 # Dictionary
testMatrix = io.loadmat(testFileMNIST)                   # Dictionary

if DEBUG:
    print 50*'-'
    print trainMatrix, testMatrix

############# GET DATA ############# 
testData = np.array(testMatrix['kaggle_image'])
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
    

training_Size = [100, 200, 500, 1000, 2000, 5000, 10000, 30000, 60000]
for overall in [True, False]:
    errorRate_array = []
    for elem in training_Size:
        all_mu, all_cov, all_prior = train_gauss(shuffledData[:elem], np.array(shuffledLabels[:elem]))
            
        predicted_Digits = gauss_predict(shuffledData[50000:], all_mu,\
                                          all_cov, all_prior, overall)
        
    #     print predicted_Digits
        
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
    ax.set_title('Error Rate Vs Training Size with Overall ' + str(overall))
    ax.set_xlabel('Training Size')
    ax.set_ylabel('Error Rate')
    ax.plot(training_Size, errorRate_array)
    plt.savefig("./Results/ErrorRate_TrainingSize_Overall_" + str(overall) + ".png")
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


############# Cross Validation ############# 
small_weight = np.linspace(0.01, 0.1, 5)
gauss_params = train_gauss(shuffledData[0:10000], shuffledLabels[0:10000])
for weight in small_weight:
    scores = computeCV_Score(gauss_params, np.array(crossValidation_Data), crossValidation_Labels, k, weight)
    scoreBuffer.append((scores).mean())
    print weight, scoreBuffer
    if DEBUG:
        print "Weight:", weight, "Accuracy: %0.2f (+/- %0.2f)" % ((scores).mean(), np.array(scores).std() / 2)
        print 50*'-'
 
maxScore = np.max(np.array(scoreBuffer))
maxScore_Index = scoreBuffer.index(maxScore)
 
print "Using Custom CV Function"
print "Best Weight:", small_weight[maxScore_Index], "Accuracy for that weight:", maxScore
print 50*'-'

#########################################################
# FOR KAGGLE
#########################################################
all_mu, all_cov, all_prior = gauss_params
indices = np.array(range(1,len(testData_flat)+1))
kaggle_format =  np.vstack(((indices), (gauss_predict(testData_flat,all_mu, all_cov, all_prior, False, small_weight[maxScore_Index])))).T

np.savetxt("./Results/Digits.csv", kaggle_format, delimiter=",", fmt = '%d,%d',   header = 'Id,Category', comments='') 

print 5*"-The End-"