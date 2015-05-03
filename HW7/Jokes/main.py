'''
Created on May 3, 2015

@author: antonio
'''
############# IMPORTS ############# 
import numpy as np
from scipy import io

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

print np.shape(jokeData)

validationData = []
for line in vf.readlines():
    validationData.append(map(int, line.strip().split(",")))
    
testData = []
for line in tf.readlines():
    testData.append(map(int, line.strip().split(",")))

print np.shape(testData), np.shape(validationData)

############# SIMPLE SYS ############# 
# Average score
av_score = []
for i in range(np.shape(jokeData)[1]):
    av_score.append(np.mean([x[i] for x in jokeData if not np.isnan(x[i])]))
    
accuracy = 0
for elem in validationData:
    if elem[2]*av_score[elem[1]-1] > 0 or (elem[2]==0 and av_score[elem[1]-1] < 0):
        accuracy+=1

print "Simple Accuracy:", np.around(100.0*accuracy/len(validationData)), "%"

############# PERSONAL PREF ############# 
