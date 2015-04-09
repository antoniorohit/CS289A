from Decision_Tree import DTree
import numpy as np
from scipy import io
import random

############# FUNCTIONS ############# 
# Entropy impurity
def entropy_impurity(left_label_hist, right_label_hist):
    """Describe"""
#     left_label_hist = (left_0_labels, left_1_labels)      
#     right_label_hist = (right_0_labels, right_1_labels)

    total_left = sum(left_label_hist)
    total_right = sum(right_label_hist)
    
    small_value = 10**-50       # avoid divide by 0
    left_0_label = small_value + left_label_hist[0]/float(total_left)
    left_1_label = small_value + left_label_hist[1]/float(total_left)

    right_0_label = small_value + right_label_hist[0]/float(total_right)
    right_1_label = small_value + right_label_hist[1]/float(total_right)

    P_left = left_0_label*np.log2(left_0_label) + left_1_label*np.log2(left_1_label)
    P_right = right_0_label*np.log2(right_0_label) + right_1_label*np.log2(right_1_label)
    
    # Average impurity
    return -(P_left + P_right)*0.5

def gini_impurity(left_label_hist, right_label_hist):
    """Describe"""
#     left_label_hist = (left_0_labels, left_1_labels)      
#     right_label_hist = (right_0_labels, right_1_labels)

    total_left = sum(left_label_hist)
    total_right = sum(right_label_hist)
    
    left_0_label = left_label_hist[0]/float(total_left)
    left_1_label = left_label_hist[1]/float(total_left)

    right_0_label = right_label_hist[0]/float(total_right)
    right_1_label = right_label_hist[1]/float(total_right)

    P_left = left_0_label*left_1_label
    P_right = right_0_label*right_1_label
    
    # Average impurity
    return (P_left + P_right)*0.5


def segmentor(data, labels, impurity):
    """Describe"""
    # Assuming that the first element of shape is sample count, 
    # second element is features
    data_len, num_features = np.shape(data)
    min_impurity_score = 1.
    min_imp_feature_ind = -1
    min_imp_threshold = -1
    
    root_1_labels = sum(labels)
    root_0_labels = data_len - root_1_labels
    
    for i in range(num_features):
        threshold = np.mean(data,axis=0)[i]
        multi_val_arr = [(x,l) for (x,l) in zip(data, labels) if x[i] < threshold]
                
        temp_data = [row[0] for row in multi_val_arr]
        temp_labels = [row[1] for row in multi_val_arr]
        
        left_data_len = np.shape(temp_data)[0]
        left_1_labels = sum(temp_labels)
        left_0_labels = left_data_len - left_1_labels

        right_1_labels = root_1_labels - left_1_labels
        right_0_labels = root_0_labels - left_0_labels
        
        left_label_hist = (left_0_labels, left_1_labels)      
        right_label_hist = (right_0_labels, right_1_labels)
                
        # The data on the left and right should be non zero
        if left_data_len > 0 and left_data_len < data_len:
            impurity_score = impurity(left_label_hist, right_label_hist)
#             if(left_0_labels == 0 or left_1_labels == 0 or right_0_labels == 0 or right_1_labels == 0):
#                 print i, left_0_labels, left_1_labels, right_0_labels, right_1_labels
        else:
            impurity_score = 1
        
        if(impurity_score <= min_impurity_score):
            min_impurity_score = impurity_score
            min_imp_feature_ind = i
            min_imp_threshold = threshold
#             print i, threshold, impurity_score
            
    return (min_imp_feature_ind, min_imp_threshold)

def computeCV_Score(clf, data, labels, folds):
    i = 0
    j = 0
    accuracy = 0.0
    scores = []
    # For each fold trained on...
    for i in range(folds):
        # Initialize variables
        j = 0
        accuracy = 0
        clf_local = clf
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

depths = [1, 5, 25, 50, 100, 150]


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
    print "DEPTH:", depth
    clf = DTree(depth, gini_impurity, segmentor)
    scores = computeCV_Score(clf, crossValidation_Data, crossValidation_Labels, k)
    scoreBuffer.append((scores).mean())
    if 1:
        print "Depth:", depth, "Accuracy: %0.2f%% (+/- %0.2f)" % ((scores).mean(), np.array(scores).std() / 2)
        print 50*'-'

maxScore = np.max(scoreBuffer)
maxScore_Index = scoreBuffer.index(maxScore)
print "Best Depth Value:", depths[maxScore_Index], "Accuracy for that Depth:", np.around(maxScore,3)
print 50*'-'

############# FOR KAGGLE ############# 
indices = np.array(range(1,len(testData)+1))
kaggle_format =  np.vstack(((indices), (clf.predict(testData)))).T
np.savetxt("./Results/spam.csv", kaggle_format, delimiter=",", fmt = '%d,%d',   header = 'Id,Category', comments='') 

print 20*"*", "The End" ,20*"*"
