'''
Created on May 3, 2015

@author: antonio
'''
'''
Created on Apr 16, 2015

Homework 6 (Neural Nets)

@author: antonio
'''
############# IMPORTS ############# 
from utils import *
from KMeans import KMeans
import pylab as plt
import time
import numpy as np
from scipy import io

features = "raw"  # or "malik" -> for HOG features
cost = "MSE"        # or "entropy"

############# FILE STUFF ############# 
trainFileMNIST = "./mnist_data/images.mat"
    
trainMatrix = io.loadmat(trainFileMNIST)  # Dictionary

print trainMatrix

output_file = open("./Results/results_" + str(features) + "_" + cost + ".txt", "wb")
############# GET DATA ############# 
print 20 * "#", "Getting Data", 20 * "#"
imageData = np.array(trainMatrix['images'])
imageData = np.rollaxis(imageData, 2, 0)  # move the index axis to be the first 

dataShape = np.shape(imageData)
print "Image Data Shape", dataShape

imageDataFlat = []
for elem in imageData:
    imageDataFlat.append(elem.flatten())

dataShape = np.shape(imageDataFlat)
print "Image Data Flat Shape", dataShape

num_clusters = [5, 10, 20]

for cluster in num_clusters:
    print 20 * "#", "Num Clusters:", cluster, 20 * "#"
    KM = KMeans(cluster)
    KM.fit(imageDataFlat)
    visualize(KM.cluster_centers_)
    