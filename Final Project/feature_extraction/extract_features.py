'''
Created on Apr 13, 2015

Provides a clean interface to using the MFCC extraction functions

@author: antonio
'''
# coding utf-8

from base import mfcc, delta
import numpy as np
import parameters as prm

# @timing
def extractFeatures(input_signal):
    """extract features from the cleaned signal.
    
    :param cleaned signal
    :return: features list"""
    
    # compute mfcc list
    if len(input_signal)==0:
        print("cleaned signal is empty")
        return input_signal
    
    mfcc_list = mfcc(input_signal, prm.params["sample_rate"].get())    
    
#     # Cepstral Mean Normalization
#     if 1:
#         mean_mfcc = np.average(mfcc_list.T, 1)
#         std_mfcc = np.std(mfcc_list.T, 1)
#         for i in range(len(mfcc_list)):
#             mfcc_list[i] = (mfcc_list[i]-mean_mfcc)/std_mfcc

#     print np.shape(mfcc_list[i-1]), np.shape(mean_mfcc), np.shape(std_mfcc)
    N = 2
    delta_list = delta(mfcc_list, N)
    ddelta_list = delta(delta_list, N)
    
    # do not keep first coeff (energy)
    features_list=list()
    for k in range(len(mfcc_list)):
        features_list += [mfcc_list[k][1:5]]
#         features_list += [delta_list[k][1:]]
#         features_list += [ddelta_list[k][1:]]
#         features_list += [np.hstack((mfcc_list[k][1:], delta_list[k][1:], ddelta_list[k][1:]))]
        
#     print np.shape(mfcc_list), np.shape(features_list)
    
    # dont return nan 
    # TODO WHY DOES THIS HAPPEN?
    for row in features_list:
        for cell in row:
            if cell!=cell:
                print cell
                return []
    
    return features_list