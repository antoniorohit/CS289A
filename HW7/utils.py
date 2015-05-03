'''
Created on Apr 16, 2015

@author: antonio
'''
############# IMPORTS ############# 
from scipy import signal
import numpy as np
from scipy import io
import random
import cPickle as pickle
import os
import scipy.ndimage.filters as filters
import pylab as plt
 
def visualize(image_list):
    for image in image_list:
        image = np.reshape(image, (28,28))
        plt.figure()
        plt.imshow(image, cmap='gray')
    plt.show()
