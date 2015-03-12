import numpy as np
from numpy import random
from scipy import io
import pylab as plt
from enum import Enum

class ProcessingMethod:
    normalized = 0
    logarith = 1
    sign = 2
    
class GradientMethod :
    batch = 0
    stochastic = 1


############# FILE STUFF ############# 
File_Spam = "./spam.mat"
GRADIENT = GradientMethod.stochastic
reg = 0.001
step_size = 0.000001
NUM_ITER = 1000

trainMatrix = io.loadmat(File_Spam)                 # Dictionary

############# GET DATA ############# 
training_data_raw = np.array(trainMatrix['Xtrain'])
trainingLabels_raw = np.array(trainMatrix['ytrain'])
testData = np.array(trainMatrix['Xtest'])

for PRE in [0, 1, 2]:
    ############# PRE-PROCESSING ############# 
    if PRE == ProcessingMethod.normalized:
        print 30*'#'
        print "Mean=0, STD=1 Features"
        step_size = 0.00005
        trainingData = (training_data_raw-np.mean(training_data_raw))/np.std(training_data_raw)
        # print np.mean(trainingData), np.std(trainingData)
        
    elif PRE == ProcessingMethod.logarith:
        print 30*'#'
        print "Log Features"
        step_size = 0.0000001
        trainingData = np.log(training_data_raw + 0.01)
        
    else:
        print 30*'#'
        print "x>0? Features"
        step_size = 0.00001
        i = 0
        j = 0
        for sample in training_data_raw:
            j = 0
            for feature in sample:
                if feature > 0:
                    trainingData[i][j] = 1
                else:
                    trainingData[i][j] = 0
                j += 1
            i += 1
    
    ############# SHUFFLE DATA ############# 
    trainingComplete = zip(trainingData, trainingLabels_raw)
    
    
    random.shuffle(trainingComplete)
    shuffledData = []
    shuffledLabels = []
    for elem in trainingComplete:
        shuffledData.append((elem[0]))                # Use a simple array of pixels as the feature
        shuffledLabels.append((elem[1]))
    
    trainingData = np.array(shuffledData)
    trainingLabels = np.array(shuffledLabels)
    
    
    X = np.matrix(trainingData)
    mu = []
    beta = np.matrix(np.zeros((57,1)))
    y = np.matrix(trainingLabels)
    loss_array = []
    
    i = 0
    for elem in trainingData:
        elem = np.matrix(elem).T
        mu.append(float((1/(1+np.exp(-beta.T*elem))))) 
     
    mu = np.matrix(mu).T
     
    if(GRADIENT == GradientMethod.batch):
    ############# BATCH GRADIENT DESCENT METHOD ############# 
        for i in range(NUM_ITER):
            loss = reg*np.square(np.linalg.norm(beta)) - y.T*np.log(mu+0.000001) - (1-y).T*np.log(1-mu + 0.000001)
            beta = beta - step_size*np.matrix(2*reg*beta - X.T*(y-mu))  
            mu = []
            for elem in trainingData:
                elem = np.matrix(elem).T
                try:
                    mu.append(float((1/(1+np.exp(-beta.T*elem))))) 
                except:
                    print beta.T*elem
         
            mu = np.matrix(mu).T
            loss_array.append(float(loss))
    elif GRADIENT == GradientMethod.stochastic:
        ############ STOCHASTIC GRADIENT METHOD ############# 
        for i in range(NUM_ITER):
            step_size_loc = (step_size)/(i+1.0)
            loss = reg*np.square(np.linalg.norm(beta)) - y.T*np.log(mu+0.000001) - (1-y).T*np.log(1-mu + 0.000001)
            beta = beta - step_size_loc*np.matrix(2*reg*beta - X[i].T*(y[i]-mu[i]))  
            mu = []
            for elem in trainingData:
                elem = np.matrix(elem).T
                try:
                    mu.append(float((1/(1+np.exp(-beta.T*elem))))) 
                except:
                    print beta.T*elem
         
            mu = np.matrix(mu).T
            loss_array.append(float(loss))
    else:
        ############ NEWTON METHOD ############# 
        for i in range(NUM_ITER):
            loss = reg*np.linalg.norm(beta) - y.T*np.log(mu) - (1-y).T*np.log(1-mu)
            beta = beta + step_size*(np.matrix(2*reg*np.ones((57, 57)) - X.T*np.matrix(np.diag(np.ravel(np.multiply(mu, (np.ones((len(trainingData),1))-mu))), 0)*X))**-1)*np.matrix(2*reg*beta - X.T*(y-mu))     
            mu = []
            for elem in trainingData:
                elem = np.matrix(elem).T
                try:
                    mu.append(float((1/(1+np.exp(-beta.T*elem))))) 
                except:
                    print beta.T*elem
        
            mu = np.matrix(mu).T
            loss_array.append(float(loss))
     
         
    print "Loss Array: ", (loss_array)
    
    plt.figure()
    plt.plot(loss_array)
    plt.savefig('./Results/Training_Loss_Stochastic_Variable_' + str(PRE))

print 30*'#'
print "THE END!"
print 30*'#'
