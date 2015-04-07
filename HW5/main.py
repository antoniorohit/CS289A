from Decision_Tree import DTree
import numpy as np
from scipy import io
import random

############# FUNCTIONS ############# 
# Entropy impurity
def impurity(left_label_hist, right_label_hist):
    """Describe"""
    total = left_label_hist + right_label_hist
    P_left = left_label_hist/total
    P_right = right_label_hist/total
    return -(P_left*np.log2(P_left) + P_right*np.log2(P_right))

def segmentor(data, labels, impurity):
    """Describe"""
    # Assuming that the first element of shape is sample count, 
    # second element is features
    data_len, num_features = np.shape(data)
    min_impurity_score = 1
    min_imp_feature_ind = -1
    min_imp_threshold = -1
    
    for i in range(num_features):
        threshold = np.mean(data,axis=1)[i]
        left_label_hist = len([x for x in data if x[i]>threshold])
        right_label_hist = data_len - left_label_hist
        impurity_score = impurity(left_label_hist, right_label_hist)
        if(impurity_score < min_impurity_score):
            min_impurity_score = impurity_score
            min_imp_feature_ind = i
            min_imp_threshold = threshold
    
    return (min_imp_feature_ind, min_imp_threshold)

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

        clf_local.train(data[i], labels[i])
        # For each validation performed (k-1 total) on a fold
        for j in range(folds):
            if(j!=i):
                predicted_Class = clf_local.predict(data[j])
                for (elem1, elem2) in zip(predicted_Class, labels[j]):
                    if elem1 == elem2:
                        accuracy+=1                
            j+=1
        scores.append(100.0*accuracy/((folds-1)*len(predicted_Class)))
        i+=1
    return np.array(scores)

############# CONSTANTS ############# 

depths = [50, 150, 200]


############# FILE STUFF ############# 
File_Spam = "./spam_data.mat"

trainMatrix = io.loadmat(File_Spam)                 # Dictionary

############# GET DATA ############# 
training_data_raw = np.array(trainMatrix['training_data'])
training_labels_raw = np.array(trainMatrix['training_labels']).T
testData = np.array(trainMatrix['test_data'])

trainingComplete = zip(training_data_raw, training_labels_raw)

############# SHUFFLE DATA ############# 
random.shuffle(trainingComplete)
trainingData = []
trainingLabels = []
for elem in trainingComplete:
    trainingData.append(elem[0]) 
    trainingLabels.append(elem[1][0])

# spam_DTree = DTree(depths[1], impurity, segmentor)
# spam_DTree.train(trainingData, trainingLabels)

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

if 0:
    print "Lengths of CV Data and Labels: ", np.array(crossValidation_Data).shape, np.array(crossValidation_Labels).shape
    print 50*'-'

scoreBuffer = []

############# CROSS-VALIDATION ############# 
for depth in depths:
    clf = DTree(depths[1], impurity, segmentor)
    scores = computeCV_Score(clf, crossValidation_Data, crossValidation_Labels, k)
    scoreBuffer.append((scores).mean())
    if 0:
        print "Depth:", depth, "Accuracy: %0.2f (+/- %0.2f)" % ((scores).mean(), np.array(scores).std() / 2)
        print 50*'-'

maxScore = np.max(scoreBuffer)
maxScore_Index = scoreBuffer.index(maxScore)
print "Best C Value:", depths[maxScore_Index], "Accuracy for that Depth:", maxScore
print 50*'-'

############# FOR KAGGLE ############# 
indices = np.array(range(1,len(testData)+1))
kaggle_format =  np.vstack(((indices), (clf.predict(testData)))).T
