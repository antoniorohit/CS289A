'''
Created on May 4, 2015

@author: antonio
'''
import numpy as np

def PCA(X, d):
    X = np.matrix(X)
    U, S, V = np.linalg.svd(X, full_matrices=False)
    return np.dot(U[:,:d],np.diag(S[:d])), V[:d]