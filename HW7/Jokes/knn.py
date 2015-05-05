'''
Created on May 4, 2015

@author: antonio
'''
import numpy as np

class KNN(object):
    def __init__(self, k):
        self.k = k
    
    def fit(self, X):
        X = np.array(X)
        dataLen = len(X)
        self.neighbours = np.zeros((dataLen, self.k))
        for i in range(100):
            distanceBuf = []
            for j in range(dataLen):
                distanceBuf.append(np.sum(np.square(X[i]-X[j])))
            sortedDistanceBuf = sorted(distanceBuf)
            self.neighbours[i]=[distanceBuf.index(x) for x in sortedDistanceBuf[:self.k]]
            
            
            
        