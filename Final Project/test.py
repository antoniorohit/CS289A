import sys

sys.path.append("./feature_extraction")

import os
import parameters as prm
from utils import *
import cPickle as pickle
from sklearn import svm

data_directory = prm.params["data_directory"].get()
prm.params["chunk_size"].set(1)
            
try:
    print "Loading the Data... (may take a while)"
    data = pickle.load(open(data_directory+"data.p", "rb"))
    labels = pickle.load(open(data_directory+"labels.p", "rb"))
    print "Data Loaded, Good to Go!"
except Exception as excp:
    print "Exception:", excp
    data, labels = extract_Data(data_directory)
    if 1:
        print "Flattening ze Data"
        # flatten the data (for svm)
        data_flat = []
        data = np.array(data)
        for elem in data:
            data_flat.append(elem.flatten())
        data = data_flat

    print "Saving the Data... (may take a while)"
    pickle.dump(data, open(data_directory+"data.p", "wb"))        
    pickle.dump(labels, open(data_directory+"labels.p", "wb"))        
    print "Data Saved, Good to Go!"


print "Shapes of Data and Labels", np.shape(data), np.shape(labels)
print "Sum of labels:", sum(labels)

# Check for nan
for elem in data:
    for cell in elem:
        if cell != cell:
            print cell
        

print "Select Data Subset... (may take a while)"
data, labels = select_Data_Subset(data, labels, fraction=0.1)
print "Data Subset Selected!"


print "Shapes of Data and Labels", np.shape(data), np.shape(labels)
print "Sum of labels:", sum(labels)

#########################################################
# CROSS VALIDATION 
#########################################################
print 50 * '='
print "CROSS VALIDATION USING SVM"
print 50 * '='

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

C = np.linspace(0.1,10,16)                   # array of values for parameter C
############# CROSS-VALIDATION ############# 
for C_Value in C:
    print "C:", np.around(C_Value, 1)
    clf = svm.SVC(kernel='linear', C=C_Value)
    scores = computeCV_Score(clf, crossValidation_Data, crossValidation_Labels, k)
    scoreBuffer.append((scores).mean())
    print "C:", np.around(C_Value, 1), "Accuracy: %0.2f%% (+/- %0.2f)" % ((scores).mean(), np.array(scores).std() / 2)
    print 50 * '-'


maxScore = np.max(scoreBuffer)
maxScore_Index = scoreBuffer.index(maxScore)
print "Best C Value:", C[maxScore_Index], "Accuracy for that C:", np.around(maxScore, 3)
print 50 * '-'



print 20*"*", "The End", 20*"*"

