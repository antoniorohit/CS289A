'''
Created on May 3, 2015

@author: antonio
'''
import numpy as np

class KMeans(object):
    def __init__(self,n_clusters=5, init='k-means++', n_init=10, max_iter=300, tol=0.0001):
        self.n_clusters = n_clusters
        self.cluster_centers_ = []
    
    def fit(self, X):
        self.dataShape = np.shape(X)
        # random initialization
        for _ in range(self.n_clusters):
            self.cluster_centers_.append(np.random.randint(256, high=None, size=self.dataShape))
        
        pass
    
    def predict(self, X):
        pass
    
    
        