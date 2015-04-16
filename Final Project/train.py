import sys

sys.path.append("./feature_extraction")

import os
import parameters as prm
from utils import *
import cPickle as pickle
from sklearn import svm, tree, ensemble
import random

data_directory = prm.params["data_directory"].get()
pickle_directory = prm.params["pickle_directory"].get()
prm.params["chunk_size"].set(.25)
            
try:
    print "Loading the Data... (may take a while)"
    data = pickle.load(open(pickle_directory+"data.p", "rb"))
    labels = pickle.load(open(pickle_directory+"labels.p", "rb"))
    print "Data Loaded, Good to Go!"
except Exception as excp:               # data.p doesnt exist
    print "Exception:", excp
    data, labels = extract_Data(data_directory)
    print "Flattening ze Data"
    # flatten the data (for svm)
    data_flat = []
    data = np.array(data)
    for elem in data:
        data_flat.append(elem.flatten())
    data = data_flat
    data_male = []
    data_female = []
    
    for elem in zip(data, labels):
        if elem[1] == 1: # male
            data_male.append(elem[0])
        else:
            data_female.append(elem[0])
        
    print "Select Data Subset... (may take a while)"
    labels_male = np.ones(len(data_male))
    labels_female = np.zeros(len(data_female))
    data_male, labels_male = select_Data_Subset(data_male, labels_male, 0.1)
    data_female, labels_female = select_Data_Subset(data_female, labels_female, 1)
    
    print "Shapes of Data (male, female) and sum of labels", np.shape(data_male), np.shape(data_female), sum(labels)

    data = np.concatenate((data_male, data_female))
    labels = np.concatenate((labels_male, labels_female))
    print "Data Subset Selected!"

    dataComplete = zip(data, labels)
    
    # SHUFFLE THE IMAGES
    random.shuffle(dataComplete)
    
    data = []
    labels = []
    for elem in dataComplete:
        data.append(elem[0])
        labels.append(elem[1])
    
    print "Saving the Data... (may take a while)"
    pickle.dump(data, open(pickle_directory+"data.p", "wb"))        
    pickle.dump(labels, open(pickle_directory+"labels.p", "wb"))        
    print "Data Saved, Good to Go!"


    print "Shapes of Data (male, female) and Labels", np.shape(data_male), np.shape(data_female), np.shape(labels)
    print "Sum of labels:", sum(labels)

# Check for nan
# for elem in data:
#     for cell in elem:
#         if cell != cell:
#             print cell
        

print "Shapes of Data and Labels", np.shape(data), np.shape(labels)
print "Sum of labels:", sum(labels)

#########################################################
# CROSS VALIDATION 
#########################################################

############# DATA PARTIONING ############# 
crossValidation_Data = []
crossValidation_Labels = []
k = 10 
stepLength = k
for index in range(0, k):
    crossValidation_Data.append(data[index:-1:stepLength])
    crossValidation_Labels.append(labels[index:-1:stepLength])

if 0:
    print "Lengths of CV Data and Labels: ", np.array(crossValidation_Data).shape, np.array(crossValidation_Labels).shape
    print 50 * '-'

scoreBuffer = []

C = np.linspace(.1, 1000. , 4)                   # array of values for parameter C
depths = [10, 25, 50, 100]

############# FORESTS/TREES ############# 
print 50 * '='
print "CROSS VALIDATION USING SCIKIT-LEARN FORESTS"
print 50 * '='

for depth in depths:
    print "DEPTH:", depth
    clf = ensemble.RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=depth)
    scores = computeCV_Score(clf, crossValidation_Data, crossValidation_Labels, k)
    scoreBuffer.append((scores).mean())
    print "Depth:", depth, "Accuracy: %0.2f%% (+/- %0.2f)" % ((scores).mean(), np.array(scores).std() / 2)
    print 50 * '-'

maxScore = np.max(scoreBuffer)
maxScore_Index = scoreBuffer.index(maxScore)
print "Best Depth Value:", depths[maxScore_Index], "Accuracy for that Depth:", np.around(maxScore, 3)
print 50 * '-'

pickle.dump(clf, open(pickle_directory+"clf.p", "wb"))        

############# SVM ############# 
kernel = 'linear'

print 50 * '='
print "CROSS VALIDATION USING", kernel, "SVM"
print 50 * '='

scoreBuffer = []
for C_Value in C:
    print "C:", np.around(C_Value, 3)
    clf = svm.SVC(kernel=kernel, C=C_Value)
    scores = computeCV_Score(clf, crossValidation_Data, crossValidation_Labels, k)
    scoreBuffer.append((scores).mean())
    print "C:", np.around(C_Value, 3), "Accuracy: %0.2f%% (+/- %0.2f)" % ((scores).mean(), np.array(scores).std() / 2)
    print 50 * '-'


maxScore = np.max(scoreBuffer)
maxScore_Index = scoreBuffer.index(maxScore)
print "Best C Value:", C[maxScore_Index], "Accuracy for that C:", np.around(maxScore, 3)
print 50 * '-'


print 20*"*", "The End", 20*"*"

