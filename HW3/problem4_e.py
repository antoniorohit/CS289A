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
import numpy as np
from sklearn import svm
from scipy import io
import random
from scipy.stats import multivariate_normal as norm

def train_gauss(im_data, labels):
    ############# 
    # Fit gaussian to each digit
    ############# 
    all_cov = []
    all_prior = []
    all_mu = []
        
    for i in range(0,2):
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
        mu = [np.mean(data_label[j]) for j in range(0, len(data_label))]
        
        # sample covariance
        cov = np.zeros((len(data_label), len(data_label)))
        data_label = data_label.T 
        for elem in (data_label):
            cov += np.mat(elem-mu).T*np.mat(elem-mu)
        cov = cov/len(data_label)
    
    
        # prior probability 
        prior = float(len(data_label))/(len(im_data))
            
#         print prior, i
        all_cov.append(cov)
        all_prior.append(prior)
        all_mu.append(mu)
    return all_mu, all_cov, all_prior

def gauss_predict(data, all_mu, all_cov, all_prior, overall = True, weight = 0.001):
    labelled_list = []
    
    # Create PDFs
    n = []
    for label in range(0, 2):
        mu = all_mu[label]
        if overall == True:
            cov = np.mean(all_cov)
        else:
            # add small value to diag to remove singularity
            small_value = weight*np.eye(len(all_cov[0]))
            cov = all_cov[label] + small_value
        n.append(norm(mean=mu, cov=(cov)))
                
    for elem in data:
        prob_list = []
        for label in range(0, 2):
            prob_list.append(np.sum(n[label].logpdf(elem))*all_prior[label])
        labelled_list.append(prob_list.index(max(prob_list)))
    
    return labelled_list

DEBUG = False
############# FILE STUFF ############# 
File_Spam = "./spam-dataset/spam_data.mat"


trainMatrix = io.loadmat(File_Spam)                 # Dictionary

if DEBUG:
    print 50*'-'
    print trainMatrix

############# GET DATA ############# 
trainingData = np.array(trainMatrix['training_data'])
trainingLabels = np.array(trainMatrix['training_labels'][0])
testData= np.array(trainMatrix['test_data'])

############# 
# Normalization
############# 
i = 0
for element in trainingData:
    if np.linalg.norm(trainingData[i]) != 0:
        trainingData[i] = (trainingData[i]/np.linalg.norm(trainingData[i]))
    i+=1

trainingComplete = zip(trainingData, trainingLabels)

############# SHUFFLE DATA ############# 
random.shuffle(trainingComplete)
shuffledData = []
shuffledLabels = []
for elem in trainingComplete:
    shuffledData.append((elem[0]))                # Use a simple array as the feature
    shuffledLabels.append((elem[1]))

trainingData = np.array(shuffledData)
trainingLabels = np.array(shuffledLabels)
if DEBUG:
    print 50*'-'
    print ("Shapes of data and labels: ", trainingData.shape, 
                                    trainingLabels.shape, len(trainingComplete))
        

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
gauss_params = train_gauss(shuffledData[0:-1], shuffledLabels[0:-1])
small_weight = np.linspace(0.001, 0.00001, 5)
for weight in small_weight:
    scores = computeCV_Score(gauss_params, crossValidation_Data, crossValidation_Labels, k, weight)
    scoreBuffer.append((scores).mean())
    if 1:
        print "Weight:", weight, "Accuracy: %0.2f (+/- %0.2f)" % ((scores).mean(), np.array(scores).std() / 2)
        print 50*'-'

maxScore = np.max(scoreBuffer)
maxScore_Index = scoreBuffer.index(maxScore)
print "Best weight Value:", small_weight[maxScore_Index], "Accuracy for that:", maxScore
print 50*'-'

#########################################################
# FOR KAGGLE
#########################################################
all_mu, all_cov, all_prior = gauss_params
indices = np.array(range(1,len(testData)+1))
kaggle_format =  np.vstack(((indices), (gauss_predict(testData, all_mu, all_cov, all_prior, False, small_weight[maxScore_Index])))).T

# if DEBUG:
#     print kaggle_format.shape, kaggle_format
#     print indices.shape, clf.predict(testData).shape

np.savetxt("./Results/Spam.csv", (kaggle_format), fmt = '%d,%d',  delimiter=",", header='Id,Category', comments='') 

print 50*'='
print "End of File"
print 50*'='