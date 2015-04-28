import sys

sys.path.append("./feature_extraction")

import os
import parameters as prm
from utils import *
import cPickle as pickle
from sklearn import svm, tree, ensemble
import random

def train():
    voxforge_directory = prm.params["voxforge_directory"].get()
    pickle_directory = prm.params["pickle_directory"].get()
    
    data, labels, rawData = getTrainData_Pickle("test_protocol")
    
#     print "Shapes of Data and Labels", np.shape(data), np.shape(labels)
#     print "Sum of labels:", sum(labels)
    
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
    
    C = np.linspace(.01, 1000. , 4)  # array of values for parameter C
    depths = [10]
    
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
     
    ############# SVM ############# 
#     kernel = 'linear'
#        
#     print 50 * '='
#     print "CROSS VALIDATION USING", kernel, "SVM"
#     print 50 * '='
#        
#     scoreBuffer = []
#     for C_Value in C:
#         print "C:", np.around(C_Value, 3)
#         clf = svm.SVC(kernel=kernel, C=C_Value)
#         scores = computeCV_Score(clf, crossValidation_Data, crossValidation_Labels, k)
#         scoreBuffer.append((scores).mean())
#         print "C:", np.around(C_Value, 3), "Accuracy: %0.2f%% (+/- %0.2f)" % ((scores).mean(), np.array(scores).std() / 2)
#         print 50 * '-'
#        
#        
#     maxScore = np.max(scoreBuffer)
#     maxScore_Index = scoreBuffer.index(maxScore)
#     print "Best C Value:", C[maxScore_Index], "Accuracy for that C:", np.around(maxScore, 3)
#     print 50 * '-'
#       
#       
#     print 20*"*", "Train End", 20*"*"

    ############# KNN ############# 
#     print 50 * '='
#     print "CROSS VALIDATION USING KNN"
#     print 50 * '='
#         
#     scoreBuffer = []    
#     num_neighbors = [5, 20, 50, 100]
#     from sklearn.neighbors import KNeighborsClassifier
#     for num in num_neighbors:
#         print "Num:", (num)
#         clf = KNeighborsClassifier(n_neighbors=num)
#         scores = computeCV_Score(clf, crossValidation_Data, crossValidation_Labels, k)
#         scoreBuffer.append((scores).mean())
#         print "Num:", (num), "Accuracy: %0.2f%% (+/- %0.2f)" % ((scores).mean(), np.array(scores).std() / 2)
#         print 50 * '-'
#        
#        
#     maxScore = np.max(scoreBuffer)
#     maxScore_Index = scoreBuffer.index(maxScore)
#     print "Best Num Value:", num_neighbors[maxScore_Index], "Accuracy for that Num:", np.around(maxScore, 3)
#     print 50 * '-'
#       
#       
#     print 20*"*", "Train End", 20*"*"

    # Save the best CLF
#     print "Saving the CLF..."
    clf = ensemble.RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=depths[maxScore_Index])
#     clf = svm.SVC(kernel=kernel, C=C[maxScore_Index])
#     clf = KNeighborsClassifier(n_neighbors=num_neighbors[maxScore_Index])

    clf.fit(data, labels)
    pickle.dump(clf, open(pickle_directory + "clf.p", "wb"))        
#     print "Done saving the CLF!"
    
    # HOW DOES THIS DECISION TREE DO ON THE SAME DATA IT FIT ON? (should be 100% ?)
    accuracy = 0
    predicted_Labels = clf.predict(data)
    index = 0
    for (elem1, elem2) in zip(predicted_Labels, labels):
        if elem1 == elem2:
            accuracy += 1
        else:
#             print 'Decision Tree Mis-classification: predicted, actual', elem1, elem2
            write_wave((1, 2, 44100), np.int16(rawData[index]), './Errors/cleaned_audio_' + str(index) + '.wav')
        index += 1
#     print "Predicted:", len(predicted_Labels), sum(predicted_Labels)
    print "Accuracy of CLF on training data itself:", np.around(100.0 * accuracy / len(predicted_Labels), 2), "%"

