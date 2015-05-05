'''
Created on May 3, 2015

@author: antonio
'''
############# IMPORTS ############# 
import numpy as np
from scipy import io
from knn import KNN
from pca import PCA

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
print 20 * "#", "Simple System", 20 * "#"
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
print 20 * "#", "Personal Pref", 20 * "#"
jokeDataNew = jokeData
# replace nan by 0
for i in range(len(jokeData)):
    jokeDataNew[i] = [0 if np.isnan(x) else x for x in jokeData[i] ]
    
for k in [10, 100, 1000]:
    print "K Value:", k
    knn = KNN(k)
    knn.fit(jokeDataNew)
    neighbours = knn.neighbours
    av_score = []
    accuracy = 0
    for i in range(100):
        average_score = (np.mean([jokeDataNew[ind] for ind in neighbours[i]], 0))
        av_score.append(average_score)
        
    for elem in validationData:
        if (elem[2]*av_score[elem[0]-1][elem[1]-1] > 0) or (elem[2]==0 and av_score[elem[0]-1][elem[1]-1] < 0):
            accuracy+=1
    
    print "Pref Accuracy:", np.around(100.0*accuracy/len(validationData)), "%"
        
############# LATENT FACTOR ANALYSIS #############
print 20 * "#", "PCA", 20 * "#"
for d in [2, 5, 10]:
    print "Value of d:", d
    u, v = PCA(jokeDataNew, d)
    reduced = np.dot(u,v)
    MSE = np.sum(np.square(reduced-jokeDataNew))
    print "MSE:", np.around(MSE)
    
    accuracy = 0
    print np.shape(reduced)
    reduced = reduced.tolist()
    for elem in validationData:
        if (elem[2]*reduced[elem[0]-1][elem[1]-1] > 0) or (elem[2]==0 and reduced[elem[0]-1][elem[1]-1] < 0):
            accuracy+=1
    
    print "PCA Accuracy:", np.around(100.0*accuracy/len(validationData)), "%"

############# KAGGLE #############
pred = []
for elem in testData:
    pred.append(reduced[elem[0]-1][elem[1]-1] > 0)

indices = np.array(range(1, len(pred) + 1))
kaggle_format = np.vstack((indices, pred)).T
np.savetxt("./Results/jokes.csv", kaggle_format, delimiter=",", fmt='%d,%d', header='Id,Category', comments='') 

print 20 * "#", "The End !", 20 * "#"
