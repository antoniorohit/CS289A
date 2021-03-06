'''
Created on Apr 16, 2015

Good References:

Series of Videos on NNs:
https://www.youtube.com/watch?v=bxe2T-V8XRs

@author: antonio
'''
import numpy as np
from collections import deque
import time

class Digit_NN(object):
    def __init__(self, dataShape, n_hidden=200, cost="MSE"):
        self.nin = dataShape
        self.nout = 10
        self.nhidden = 200
        self.gamma = 10 ** -3
        self.yHat = []
        # initialize weights
        self.W1 = 0.0001 * np.random.randn(dataShape + 1, self.nhidden)    
        self.W2 = 0.0001 * np.random.randn(self.nhidden + 1, self.nout)
        if cost == "MSE":
            self.costFunction = self.costFunction_mse
            self.epsilon = 10 ** -5
        else:  # entropy
            self.costFunction = self.costFunction_entropy
            self.epsilon = 10 ** -1
            
    def derivative_tanh(self, z):
        """ Computes the derivative of the sigmoid of z"""
        z = np.array(z)
        return (1. - np.square(np.tanh(z)))

    def derivative_sig(self, z):
        """ Computes the derivative of the sigmoid of z"""
        z = np.array(z)
        return np.exp(-z) / (np.square(1. + np.exp(-z)))
    
    def sigmoid(self, z):
        """ Computes the sigmoid of z"""
        return (1. / (1 + np.exp(-z)))

    
    def forward(self, X):
        X = np.hstack((X, np.ones((np.size(X, 0), 1))))
        self.z2 = np.dot(X, self.W1)
        self.a2 = np.hstack((np.tanh(self.z2), np.ones((np.shape(X)[0], 1))))
        self.z3 = np.dot(self.a2, self.W2)
        self.yHat = self.sigmoid(self.z3)
    
    def backprop(self, x, y):
        x = np.hstack((x, np.ones((np.size(x, 0), 1))))
        # W2
        # a2: 1x(nhidden+1), delta3: 1xnout
        if self.costFunction == self.costFunction_mse:
            costFunc_prime = -(y - self.yHat)
            delta3 = np.multiply(costFunc_prime, self.derivative_sig(self.z3))
        else:
            # @TODO
            delta3 = -(y - self.yHat)

        dJdW2 = np.dot(self.a2.T, delta3)
        self.W2 -= dJdW2 * self.gamma
        
        # W1
        # delta2: 1x200 
        # @TODO - drop the first or last coeff of W2??? i.e. 1: or :-1
        delta2 = np.multiply(np.dot(delta3, self.W2.T)[:,:-1], self.derivative_tanh(self.z2))
        dJdW1 = np.dot(x.T, delta2)
        self.W1 -= dJdW1 * self.gamma
        
        return dJdW1, dJdW2

    
    def fit(self, data, labels):
        labels = np.array(labels)
        data = np.array(data)
        data_len = len(data)
        completeData = zip(data, labels)
        cost = deque((self.costFunction(data, labels)) * np.ones(10))
        curr_cost = np.mean(cost)
        delta = 1
        i = 0
        j = 0
        startTime = 0
        accuracy = 0

        orig_gamma = self.gamma
        while 1:
            x, y = completeData[np.random.randint(data_len, size=1)[0]]

#             numgrad = self.computeNumericalGradient(self, x, y)
            self.forward(np.matrix(x))
            dJdW1, dJdW2 = self.backprop(np.matrix(x), y)
#             grad = np.concatenate((dJdW1.ravel().T.tolist(), dJdW2.ravel().T.tolist())).ravel()
#             numer = np.linalg.norm(grad - numgrad)
#             denom = np.linalg.norm(grad + numgrad)
#             print numer
#             print denom
#             print "The factor!!", numer / denom

            if i % 1000 == 0:
#                 print "Loop Time:", time.time()-startTime
                curr_cost = (self.costFunction(data, labels))    
                old_cost = np.mean(cost)
                cost.append(curr_cost)
                cost.popleft()
                delta = np.mean(cost) - old_cost
                if(np.abs(delta) <= self.epsilon) or i > 1000000 or curr_cost <= self.epsilon or accuracy > 99.9:
                    j += 1
                    if(j >= 5 or curr_cost == 0):
                        print "Cost and Delta:", cost, delta
                        self.gamma = orig_gamma
                        break   
                else:
                    j = 0
                    
                accuracy = 0
                pred_labels = self.predict(data)
                for elem1, elem2 in zip(pred_labels, labels):
                    elem2 = elem2.tolist().index(1)
                    if elem1 == elem2:
                        accuracy += 1
                        
                accuracy = 100.0*accuracy/len(pred_labels)
                # "i, Cost, Delta, Accuracy:" 
                print i, np.around(curr_cost, 3), np.around(delta, 6), accuracy
                if i > 200000:
                    self.gamma = 100*orig_gamma / np.sqrt(i)
                        
#                 startTime = time.time()

            i += 1
        return self.W1, self.W2
    
    def predict(self, testData):
        # append bias of 1
        predicted = []
        self.forward(np.matrix(testData))
        for elem in testData:
            self.forward(np.matrix(elem))
            nn_label = np.around(self.yHat, 0).T.tolist()
            max_value = np.max(nn_label)
            max_index = nn_label.index(max_value)
#             print max_index
            predicted.append(max_index)
        
        return predicted        
    
    
    def costFunction_mse(self, x, y):
        # Compute cost for given X,y, use weights already stored in class.
        self.forward(np.matrix(x))  # this forward pass is over the entire training set. 
        J = 0.5 * np.sum(np.mean(np.square(y - self.yHat), 0).T)  # mean over all training examples, sum over outputs         
        return J 
    
    def costFunction_entropy(self, x, y):
        # Compute cost for given X,y, use weights already stored in class.
        self.forward(np.matrix(x))
        stability_term = 10 ** -8
        term1 = np.multiply(y, np.log(self.yHat + stability_term))
        term2 = np.multiply((1 - y), np.log(1 - self.yHat + stability_term))
        error_matrix = term1 + term2
        J = -np.sum(np.sum(error_matrix))
        return J

    def setParams(self, params):
        # Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.nhidden * (self.nin + 1)
        self.W1 = np.reshape(params[W1_start:W1_end], (self.nin + 1 , self.nhidden))
        W2_end = W1_end + (self.nhidden + 1) * self.nout
        self.W2 = np.reshape(params[W1_end:W2_end], (self.nhidden + 1, self.nout))


    def computeNumericalGradient(self, N, X, y):
        paramsInitial = np.concatenate((N.W1.ravel(), N.W2.ravel()))
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        epsilon = 10 ** -5

        for p in range(len(paramsInitial)):
            # Set perturbation vector
            perturb[p] = epsilon
            N.setParams(paramsInitial + perturb)
            loss2 = N.costFunction(X, y)
            
            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X, y)

            # Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2.0 * epsilon)

            # Return the value we changed to zero:
            perturb[p] = 0
            
        # Return Params to original value:
        N.setParams(paramsInitial)

        return numgrad 
    
    def softmax(self, labels):
        new_labels = []
        denominator_sm = sum(np.exp(labels))
        for i in range(10):
            new_labels.append(np.exp(labels[i]) / denominator_sm)
        max_value = max(new_labels)
        max_index = new_labels.index(max_value)
        return max_index
