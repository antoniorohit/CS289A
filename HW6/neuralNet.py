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
        # initialize weights
        self.W1 = 0.001*np.random.randn(dataShape+1, self.nhidden)    
        self.W2 = 0.001*np.random.randn(self.nhidden+1, self.nout)

    def sig_derivative(self, z):
        """ Computes the sigmoid of z"""
        return (np.exp(-z)/(1+np.exp(-z))**2)
    
    def sigmoid(self, z):
        """ Computes the sigmoid of z"""
        return (1./(1+np.exp(-z)))
    
    def forward(self, X):
        self.z2 = np.dot(X, self.W1)
        self.a2 = np.tanh(self.z2)
        self.a2 = np.hstack((self.a2, np.ones((len(self.a2),1))))
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat
    
    def fit(self, data, labels):
        return
    
    def predict(self, testData):
        # append bias of 1
        testData = np.hstack((testData, np.ones((len(testData),1))))
        return self.forward(testData)
    
    def computeNumericalGradient(self):
        return