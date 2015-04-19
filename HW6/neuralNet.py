'''
Created on Apr 16, 2015

@author: antonio
'''
import numpy as np
from collections import deque

class Digit_NN(object):
    def __init__(self, dataShape, n_hidden=200):
        self.nin = dataShape
        self.nout = 10
        self.nhidden = 200
        self.gamma = 10**-4
        # initialize weights
        self.W1 = 0.001*np.random.randn(dataShape+1, self.nhidden)    
        self.W2 = 0.001*np.random.randn(self.nhidden+1, self.nout)

    def derivative_tanh(self, z):
        """ Computes the derivative of the sigmoid of z"""
        z = np.array(z)
        return (1-(np.tanh(z))**2)

    def derivative_sig(self, z):
        """ Computes the derivative of the sigmoid of z"""
        return np.multiply(self.sigmoid(z), (1-self.sigmoid(z)))
    
    def sigmoid(self, z):
        """ Computes the sigmoid of z"""
        return (1/(1.+np.exp(-z)))
    
    def forward(self, X, W1, W2):
        self.z2 = np.dot(X, W1)
        self.a2 = np.tanh(self.z2)
        self.a2 = np.hstack((self.a2, np.ones((len(self.a2),1))))
        self.z3 = np.dot(self.a2, W2)
        self.yHat = self.sigmoid(self.z3)
    
    def backprop(self, x, y):
        # W2
        delta3 = np.multiply(-(y-self.yHat), self.derivative_sig(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        self.W2 -= dJdW2*self.gamma
        
        # W1
        delta2 = np.multiply(np.dot(delta3, self.W2.T)[:,self.nhidden-1], self.derivative_tanh(self.z2))
        dJdW1 = np.dot(x.T, delta2)
        self.W1 -= dJdW1*self.gamma
    
    
    def fit(self, data, labels):
        labels = np.array(labels)
        data = np.array(data)
        data = np.hstack((data, np.ones((len(data),1))))
        data_len = len(data)
        completeData = zip(data, labels)
        epsilon = 10**-6
        cost = deque(10000*np.ones(50))
        delta = 1
        i = 0
        j=0
        while 1:
            if(np.abs(delta) < epsilon):
                j+=1
                if(j>10):
                    print "JOY"
                    self.gamma = 10**-4
                    break   
            else:
                j=0
                
            x,y = completeData[np.random.randint(data_len, size=1)]
            self.forward(np.matrix(x), self.W1, self.W2)
            self.backprop(np.matrix(x), y)
            curr_cost = (self.costFunction(data,labels))
            curr_cost = np.mean(curr_cost)                    
                    
            delta = curr_cost - np.mean(cost)
#             if delta > 0:
#                 self.gamma /= 2
            if i %10000 == 0:
                print "Cost, Delta:", curr_cost, delta
                if curr_cost < 23:
                    self.gamma = 10**-6
                    if curr_cost < 15:
                        self.gamma = 10**-7

            cost.append(curr_cost)
            cost.popleft()
#             print "Iteration, Change:", i, delta
            i+=1
        return self.W1, self.W2
    
    def predict(self, testData, W1, W2):
        # append bias of 1
        testData = np.hstack((testData, np.ones((len(testData),1))))
        predicted = []
        self.forward(np.matrix(testData), W1, W2)
        for elem in testData:
            self.forward(np.matrix(elem), W1, W2)
            print np.around(self.yHat, 0)
            max_value = max(self.yHat)
            max_index = np.array(self.yHat).index(max_value)
            print max_index
            predicted.append(max_index)
        
        return predicted        
    
    def computeNumericalGradient(self):
        return
    
    def softmax(self, labels):
        new_labels = []
        denominator_sm = sum(np.exp(labels))
        for i in range(10):
            new_labels.append(np.exp(labels[i])/denominator_sm)
        max_value = max(new_labels)
        max_index = new_labels.index(max_value)
        return max_index
    
    def costFunction(self, x, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.forward(np.matrix(x), self.W1, self.W2)
        J = 0.5*sum(np.array(y-self.yHat)**2)
        return J