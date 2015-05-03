'''
Created on May 3, 2015

@author: antonio
'''
import numpy as np
from blaze.expr.table import Label

class KMeans(object):
    def __init__(self,n_clusters=5, init='k-means++', n_init=10, max_iter=5, tol=0.0001):
        self.n_clusters = n_clusters
        self.cluster_centers_ = []
        self.max_iter = max_iter
        
    def fit(self, X):
        self.dataShape = np.shape(X[0])
        self.labels = np.zeros(len(X))
        # random initialization
#         for _ in range(self.n_clusters):
#             self.cluster_centers_.append(np.random.randint(256, high=None, size=self.dataShape))
        self.cluster_centers_ = [X[i] for i in np.random.choice(range(len(X)), size=10, replace=False)]       
        
        
        for iteration in range(self.max_iter):
            print "Num Iter:", iteration
            i=0
            # calculate clusters
            for elem in X:
                scoreBuf = []
                for cluster in self.cluster_centers_:
                    temp = (np.matrix(elem-cluster))
                    scoreBuf.append(np.dot(temp, temp.T))
                self.labels[i] = np.argmin(scoreBuf)
                i+=1
            
            print np.shape(self.labels)
            
            # recompute means
            for i in range(self.n_clusters):
                self.cluster_centers_[i] = np.mean([x[0] for x in zip(X, self.labels) if x[1] == i], 0)
                print np.shape(([x[0] for x in zip(X, self.labels) if x[1] == i]))
                print np.shape(np.mean([x[0] for x in zip(X, self.labels) if x[1] == i], 0))
        
        
    def predict(self, X):
        pass
    
    
        