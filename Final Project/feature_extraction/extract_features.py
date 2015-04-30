'''
Created on Apr 13, 2015

Provides a clean interface to using the MFCC extraction functions
See: https://github.com/ppwwyyxx/speaker-recognition/blob/master/doc/06-Final-Report.pdf
for some of the motivation on the numbers chosen
@author: antonio
'''
# coding utf-8
import sys
sys.path.append("./pitch_estimation/")
from base import mfcc, delta
import numpy as np
import parameters as prm
from LPCC import LPCExtractor
from brewed import extract_pitch

# @timing
def extractFeatures(input_signal):
    """extract features from the cleaned signal.
    
    :param cleaned signal
    :return: features list"""
    
    # compute mfcc list
    if len(input_signal) == 0:
        print("cleaned signal is empty")
        return input_signal
    
    mfcc_list = np.array(mfcc(input_signal, samplerate=prm.params["sample_rate"].get(), winlen=0.032, winstep=0.016, numcep=30,
          nfilt=55, nfft=2048, lowfreq=0, highfreq=6000, preemph=0.95, ceplifter=22, appendEnergy=True )) 
#     extractor = LPCExtractor(prm.params["sample_rate"].get(), 32, 16, 30, 0.95)
#     lpcc = extractor.extract(input_signal)
    
    pitch = extract_pitch(input_signal)
#     print pitch
    
#     # Cepstral Mean Normalization @TODO: WHY IS THIS NOT HELPING??
    if 0:
        mean_mfcc = np.mean(mfcc_list.T, 1)
        std_mfcc = np.std(mfcc_list.T, 1)
        for i in range(len(mfcc_list)):
            for j in range(len(mfcc_list[i])):
                mfcc_list[i][j] = (mfcc_list[i][j]-mean_mfcc[j])/std_mfcc[j]
 
#     print np.shape(mfcc_list[i-1]), np.shape(mean_mfcc), np.shape(std_mfcc)
    N = 2
    delta_list = delta(mfcc_list, N)
    ddelta_list = delta(delta_list, N)
    
    # do not keep first coeff (energy)
    features_list = list()
    for k in range(len(mfcc_list)):
#         features_list += [np.hstack((mfcc_list[k][0:], lpcc[k][0:]))]
        features_list += [mfcc_list[k][0:]]
#         features_list += [lpcc[k][0:]]
#         features_list += [np.hstack((mfcc_list[k][0:], delta_list[k][0:], ddelta_list[k][0:]))]
        
#     print np.shape(mfcc_list), np.shape(features_list)
    
    # dont return nan 
    # @TODO WHY DOES THIS HAPPEN?
    for row in features_list:
        for cell in row:
            if cell != cell:
                print "Cell is nan (see feature extraction):", str(cell)
                return []
    
    features_list = []
    features_list = list(np.ravel(features_list))
    features_list.append(pitch)         # do we need to append this multiple times to ensure that the forest selects it?
#     features_list.append(pitch)         # do we need to append this multiple times to ensure that the forest selects it?
#     features_list.append(pitch)         # do we need to append this multiple times to ensure that the forest selects it?
#     features_list.append(pitch)         # do we need to append this multiple times to ensure that the forest selects it?
#     features_list.append(pitch)         # do we need to append this multiple times to ensure that the forest selects it?
#     features_list.append(pitch)         # do we need to append this multiple times to ensure that the forest selects it?
#     features_list.append(pitch)         # do we need to append this multiple times to ensure that the forest selects it?
#     features_list.append(pitch)         # do we need to append this multiple times to ensure that the forest selects it?
#     features_list.append(pitch)         # do we need to append this multiple times to ensure that the forest selects it?
#     features_list.append(pitch)         # do we need to append this multiple times to ensure that the forest selects it?
    
    return features_list
