'''
Created on May 3, 2015

@author: antonio
'''
import numpy as np
from blaze.expr.table import Label

class KMeans(object):
    def __init__(self,n_clusters=5, init='k-means++', n_init=10, max_iter=10, tol=0.0001):
        self.n_clusters = n_clusters
        self.cluster_centers_ = []
        self.max_iter = max_iter
        self.loss = 0
        
    def fit(self, X):
        self.dataShape = np.shape(X[0])
        self.labels = np.zeros(len(X))
        # random initialization
        self.cluster_centers_ = [X[i] for i in np.random.choice(range(len(X)), size=self.n_clusters, replace=False)]       
        
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
                        
            # recompute means
            for j in range(self.n_clusters):
                self.cluster_centers_[j] = np.mean([x[0] for x in zip(X, self.labels) if x[1] == j], 0)
                print "Shape of Points in Centroids:", np.shape(([x[0] for x in zip(X, self.labels) if x[1] == j]))
        
            self.loss = 0
            # compute loss
            for ind in range(len(X)):
                mean_ = self.cluster_centers_[int(self.labels[ind])] 
                temp = (np.matrix(X[ind]-mean_))
                self.loss += (np.dot(temp, temp.T))
            self.loss =  np.sqrt(self.loss/len(X))
            print "Loss:", int(self.loss)

            
    def predict(self, X):
        return self.labels
    
    
        