'''
Created on Apr 16, 2015

@author: antonio
'''
import numpy as np


class Digit_NN(object):
    def __init__(self, dataShape, n_hidden=200):
        self.nin = dataShape
        self.nout = 10
        self.nhidden = 200
        self.gamma = 0.5
        # initialize weights
        self.W1 = 0.001*np.random.randn(dataShape+1, self.nhidden)    
        self.W2 = 0.001*np.random.randn(self.nhidden+1, self.nout)

    def sig_derivative(self, z):
        """ Computes the derivative of the sigmoid of z"""
        return np.multiply(self.sigmoid(z), (1-self.sigmoid(z)))
    
    def sigmoid(self, z):
        """ Computes the sigmoid of z"""
        return (1/(1.+np.exp(-z)))
    
    def forward(self, X):
        self.z2 = np.dot(X, self.W1)
        self.a2 = np.tanh(self.z2)
        self.a2 = np.hstack((self.a2, np.ones((1,1))))
        self.z3 = np.dot(self.a2, self.W2)
        self.yHat = self.sigmoid(self.z3)
        return self.yHat
    
    def backprop(self, x, y):
        self.yHat = self.forward(x)
        # W2
        delta3 = np.multiply(-(y-self.yHat), self.sig_derivative(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        self.W2 -= dJdW2*self.gamma
        
        # W1
        delta2 = np.multiply(np.dot(delta3, self.W2.T)[:,self.nhidden-1], self.sig_derivative(self.z2))
        dJdW1 = np.dot(x.T, delta2)
        self.W1 -= self.gamma*dJdW1
        return dJdW1, dJdW2
    
    
    def fit(self, data, labels):
        labels = np.array(labels)
        data = np.array(data)
        data = np.hstack((data, np.ones((len(data),1))))
        for (x,y) in zip(data,labels):
            self.backprop(np.matrix(x), y)
        return
    
    def predict(self, testData):
        # append bias of 1
        testData = np.hstack((testData, np.ones((len(testData),1))))
        predicted = []
        for elem in testData:
            predicted.append(self.softmax(self.forward(np.matrix(elem)).T))
        
        return predicted
    
    
    def update_weights(self, x, z, y):
        der_sig = self.sig_derivative(z)
        
    
    def computeNumericalGradient(self):
        return
    
    def softmax(self, labels):
        new_labels = []
        denominator_sm = sum(np.exp(labels))
        for i in range(10):
            new_labels.append(np.exp(labels[i])/denominator_sm)
        max_value = max(new_labels)
        max_index = new_labels.index(max_value)

        print max_index
        return max_index
    
    def costFunction(self, x, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(x)
        J = 0.5*sum((y-self.yHat)**2)
        return J