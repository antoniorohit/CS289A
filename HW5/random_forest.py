from Decision_Tree import DTree
import numpy as np
from scipy import io
import random
from sklearn import tree
from main import segmentor, entropy_impurity, gini_impurity, load_data

def create_forest_data(trainingData, trainingLabels):
    trainingComplete = np.array(zip(trainingData, trainingLabels))
    data_len = len(trainingData)
    indices = np.random.randint(data_len,size=data_len)
    trainingComplete_replaced = trainingComplete[indices,:]
    trainingData = trainingComplete_replaced[:,0].tolist()
    trainingLabels = trainingComplete_replaced[:,1].tolist()
    print "Created Forest Data"
    return trainingData, trainingLabels

def create_forest(clf, trainingData, trainingLabels, NUM_TREES=50):
    classifier_list = []
    for _ in range(NUM_TREES):
        forestData, forestLabels = create_forest_data(trainingData, trainingLabels)
        clf_new = clf
        clf_new.fit(forestData, forestLabels)
        classifier_list.append(clf_new)
    return classifier_list

def predict_forest(classifier_list, testDatum):
    label = 0.
    for clf in classifier_list:
        label += int(clf.predict([testDatum,])[0])
        
    return label/len(classifier_list)

DEPTH = 5
NUM_TREES = 50

############# FILE STUFF ############# 
File_Spam = "./Data/spam_data.mat"

trainingData, trainingLabels, testData = load_data(File_Spam)

############# CREATE FOREST ############# 
clf = DTree(DEPTH, gini_impurity, segmentor)
classifier_list = create_forest(clf, trainingData, trainingLabels, NUM_TREES)

############# PREDICT! ############# 
predictedClass = []
for elem in trainingData:
    predictedClass.append(predict_forest(classifier_list, elem))


############# ACCURACY ############# 
accuracy = 0
# For each validation performed (k-1 total) on a fold
for (elem1, elem2) in zip(predictedClass, trainingLabels):
    if elem1 == elem2:
        accuracy+=1                

print 100.0*accuracy/len(predictedClass)
print sum(predictedClass), len(predictedClass)

# print predictedClass

print 20*"*", "The End" ,20*"*"
