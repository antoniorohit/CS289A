'''
Created on Apr 13, 2015

This file contains helper functions for the male female 
detector

@author: antonio
'''
import wave
import numpy as np
import parameters as prm
import os
from extract_features import extractFeatures

def get_RawSignal(file_name):
    '''Opens filename, and returns (params, rawSignal)'''
    wf = wave.open(file_name, 'rb')
    n = wf.getnframes()
    string = wf.readframes(n)
    params = [wf.getnchannels(), wf.getsampwidth(),
               wf.getframerate(), wf.getnframes(),
               wf.getcomptype(), wf.getcompname()]
    rawSignal = np.fromstring(string, np.int16)
    return params, rawSignal

def frame_chunks(rawSignal):
    '''Breaks rawSignal into chunk_size blocks, RETURNS the framedSignal'''
    chunk_size = prm.params["chunk_size"].get()*prm.params["sample_rate"].get()
    framedRawSignal = []
    index = 0
    while index < (len(rawSignal) - chunk_size):
        framedRawSignal.append(rawSignal[index:index+chunk_size])
        index += chunk_size             #non-overlapping
    
    # Zero padding for last sample (unnecessary?)    (remove)
#     if index != len(rawSignal):
#         framedRawSignal.append(rawSignal[index:])
#         framedRawSignal[-1] = np.append(framedRawSignal[-1], np.zeros(chunk_size - len(framedRawSignal[-1])))

    return framedRawSignal

def write_wave(params, cleaned_signal, output_fname = 'cleaned_audio.wav', output_dir = ''):
    '''Creates a wave file from an input signal array'''
    wf = wave.open(output_dir+output_fname, 'wb')
    wf.setnchannels(params[0])
    wf.setsampwidth(params[1])
    wf.setframerate(params[2])
    wf.writeframes(bytearray(cleaned_signal))
    wf.close()





def extract_Gender_Label(directory):
    folder = directory+"/etc/"
    readme =  open(folder+"README").read().strip().lower()
    if("female" in readme):
        label = 0
    elif("male" in readme):
        label = 1
    else:
        print "ERROR!!"
        print readme
        exit
    return label

def extract_Data(data_directory):
    labels = []
    data = []
    for folder in os.listdir(data_directory):
        if "." not in folder and "an4" not in folder and "anonymous" not in folder:
            print folder
            directory = data_directory+folder
            folder = "/wav/"
            try:
                for file_name in os.listdir(directory+folder):
                    if file_name[-3:] == "wav":
                        label = extract_Gender_Label(directory)
                        _, rawSignal = get_RawSignal(directory+folder+file_name)
                        framedRawSignal = (frame_chunks(rawSignal))
                        for chunk in framedRawSignal:
                            feature_list = extractFeatures(chunk)
                            if feature_list != []:                          
                                data.append(feature_list)
                                labels.append(label)
#                         print np.shape(data), np.shape(data[-1]), np.shape(labels)
                        
            except OSError, e:
                print "OS Error:", str(e)
    return data, labels


def select_Data_Subset(data, labels, fraction):
    # It works - I checked
    trainingComplete = np.array(zip(data, labels))
    data_len = len(data)
    indices = np.random.randint(data_len, size=int(data_len*fraction))
    trainingComplete_replaced = trainingComplete[indices, :]
    trainingData = trainingComplete_replaced[:, 0].tolist()
    trainingLabels = trainingComplete_replaced[:, 1].tolist()
    return trainingData, trainingLabels






def computeCV_Score(clf, data, labels, folds):
    ###################################
    # Function to calculate cross validation scores 
    # Input: SVC object, data, labels, num folds
    # Output: Array of scores averaged for each fold
    ###################################    i = 0
    j = 0
    accuracy = 0.0
    scores = []
    clf_local = clf
    # For each fold trained on...
    print "Actual   :", len(labels[0]), sum(labels[0])             
    for i in range(folds):
        # Initialize variables
        clf_local = clf
        j = 0
        accuracy = 0

        try:
            clf_local.fit(data[i], labels[i])
            # For each validation performed (k-1 total) on a fold
            for j in range(folds):
                if(j!=i):
                    predicted_Labels = clf_local.predict(data[j])
                    for (elem1, elem2) in zip(predicted_Labels, labels[j]):
                        if elem1 == elem2:
                            accuracy+=1
                        else:
                            pass
                j+=1
            print "Predicted:", len(predicted_Labels), sum(predicted_Labels)
            scores.append(100.0*accuracy/((folds-1)*len(predicted_Labels)))
            i+=1
        except Exception, e:
            print str(e)
    return np.array(scores)
                

