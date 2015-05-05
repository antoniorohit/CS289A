'''
Created on May 3, 2015

@author: antonio
'''
############# IMPORTS ############# 
import numpy as np
from scipy import io
from knn import KNN

############# FILE STUFF ############# 
trainFile = "./joke_data/joke_train.mat"
validationFile = "./joke_data/validation.txt"
testFile = "./joke_data/query.txt"

trainMatrix = io.loadmat(trainFile)  # Dictionary
vf = open(validationFile, "rb")
tf = open(testFile, "rb")

############# GET DATA ############# 
print 20 * "#", "Getting Data", 20 * "#"
jokeData = np.array(trainMatrix['train'])

print "Joke Data Shape:", np.shape(jokeData)

validationData = []
for line in vf.readlines():
    validationData.append(map(int, line.strip().split(",")))
    
testData = []
for line in tf.readlines():
    testData.append(map(int, line.strip().split(",")))

print "Validation, Test Data Shape:", np.shape(validationData), np.shape(testData)

############# SIMPLE SYS ############# 
# Average score
av_score = []
accuracy = 0
for i in range(np.shape(jokeData)[1]):
    av_score.append(np.mean([x[i] for x in jokeData if not np.isnan(x[i])]))
    
for elem in validationData:
    if elem[2]*av_score[elem[1]-1] > 0 or (elem[2]==0 and av_score[elem[1]-1] <= 0):
        accuracy+=1

print "Simple Accuracy:", np.around(100.0*accuracy/len(validationData)), "%"

############# PERSONAL PREF #############
# replace nan by 0
for i in range(len(jokeData)):
    jokeData[i] = [0 if np.isnan(x) else x for x in jokeData[i] ]
    
for k in [10, 100, 1000, 10000]:
    print "K Value:", k
    knn = KNN(k)
    knn.fit(jokeData)
    neighbours = knn.neighbours
    av_score = []
    accuracy = 0
    for i in range(100):
        av_score.append(np.mean([jokeData[ind] for ind in neighbours[i]]))
        
    for elem in validationData:
        if elem[2]*av_score[elem[1]-1] > 0 or (elem[2]==0 and av_score[elem[1]-1] <= 0):
            accuracy+=1
    
    print "Pref Accuracy:", np.around(100.0*accuracy/len(validationData)), "%"
        
     
