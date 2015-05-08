'''
Created on Apr 13, 2015

This file contains helper functions for the male female 
detector

@author: antonio
'''
import sys
sys.path.append("./VAD")

import wave
import numpy as np
import parameters as prm
import os
from extract_features import extractFeatures
import cPickle as pickle
import random
from simple import remove_silence
from aed import aeDetect
import scipy.io.wavfile


def get_RawSignal(filename):
    '''Opens filename, and returns (params, rawSignal)'''
    wf = wave.open(filename, 'rb')
    _, rawSignal = scipy.io.wavfile.read(filename)

    params = [wf.getnchannels(), wf.getsampwidth(),
               wf.getframerate(), wf.getnframes(),
               wf.getcomptype(), wf.getcompname()]
    
    # convert from stereo to mono
    if params[0] == 2:
        monoChannel = np.int16(rawSignal.mean(axis=1))
        rawSignal = monoChannel

#     print filename, params
    return params, rawSignal

def frame_chunks(rawSignal):
    '''Breaks rawSignal into chunk_size blocks, RETURNS the framedSignal'''
    chunk_size = prm.params["chunk_size"].get() * prm.params["sample_rate"].get()
    framedRawSignal = []
    index = 0
    while index <= (len(rawSignal) - chunk_size):
        framedRawSignal.append(rawSignal[index:index + chunk_size])
        index += chunk_size  # non-overlapping
    
    # Zero padding for last sample (unnecessary?)    (remove?)
    if index != len(rawSignal):
        framedRawSignal.append(rawSignal[index:])
        framedRawSignal[-1] = np.append(framedRawSignal[-1], np.zeros(chunk_size - len(framedRawSignal[-1])))

    return framedRawSignal

def write_wave(params, cleaned_signal, output_fname='cleaned_audio.wav', output_dir=''):
    '''Creates a wave file from an input signal array'''
    wf = wave.open(output_dir + output_fname, 'wb')
    wf.setnchannels(params[0])
    wf.setsampwidth(params[1])
    wf.setframerate(params[2])
    wf.writeframes(bytearray(cleaned_signal))
    wf.close()




def extract_Gender_Label(directory):
    folder = directory + "/etc/"
    readme = open(folder + "README").read().strip().lower()
    if("female" in readme):
        label = 0
    elif("male" in readme):
        label = 1
    else:
        print "Neither Male nor Female in Readme Txt!!"
        print readme
    return label

def extract_Data(data_directory):
    labels = []
    data = []
    rawData = []
    for folder in os.listdir(data_directory):
        if "." not in folder and "an4" not in folder and "anonymous" not in folder:
            print folder
            directory = data_directory + folder
            folder = "/wav/"
            try:
                for file_name in os.listdir(directory + folder):
                    if file_name[-3:] == "wav":
                        label = extract_Gender_Label(directory)
                        params, rawSignal = get_RawSignal(directory + folder + file_name)
                        prm.params["sample_rate"].set(params[2])
#                         fs, cleanSignal = remove_silence(params[2], rawSignal)
                        cleanSignal = aeDetect(rawSignal)
                        framedRawSignal = (frame_chunks(cleanSignal))
                        for chunk in framedRawSignal:
                            feature_list = extractFeatures(chunk)
                            if feature_list != []:                          
                                data.append(feature_list)
                                labels.append(label)
                                rawData.append(chunk)
#                         print np.shape(data), np.shape(data[-1]), np.shape(labels)
                        
            except OSError, e:
                print "OS Error:", str(e)
    return data, labels, rawData


def select_Data_Subset(data, labels, fraction):
    # It works - I checked
    trainingComplete = np.array(zip(data, labels))
    data_len = len(data)
    indices = np.random.randint(data_len, size=int(data_len * fraction))
    trainingComplete_replaced = trainingComplete[indices, :]
    trainingData = trainingComplete_replaced[:, 0].tolist()
    trainingLabels = trainingComplete_replaced[:, 1].tolist()
    return trainingData, trainingLabels



def getTrainData_Pickle(method="voxforge"):
    if method == "voxforge":
        data, labels, rawData = getData_Voxforge()
    else:  # test_protocol
        data, labels, rawData = getData_TestProtocol("train")
    
    return data, labels, rawData

def getData_Voxforge():
    pickle_directory = prm.params["pickle_directory"].get()
    voxforge_directory = prm.params["voxforge_directory"].get()

    try:
        print "Loading the Data... (may take a while)"
        data = pickle.load(open(pickle_directory + "data_vf.p", "rb"))
        labels = pickle.load(open(pickle_directory + "labels_vf.p", "rb"))
        print "Data Loaded, Good to Go!"
    except Exception as excp:  # data_vf.p doesnt exist
        print "Exception:", excp
        data, labels, rawData = extract_Data(voxforge_directory)
        print "Flattening ze Data"
        # flatten the data (for svm)
        data_flat = []
        data = np.array(data)
        for elem in data:
            data_flat.append(elem.flatten())
        data = data_flat
        data_male = []
        data_female = []
        
        for elem in zip(data, labels):
            if elem[1] == 1:  # male
                data_male.append(elem[0])
            else:
                data_female.append(elem[0])
            
        print "Select Data Subset... (may take a while)"
        labels_male = np.ones(len(data_male))
        labels_female = np.zeros(len(data_female))
        data_male, labels_male = select_Data_Subset(data_male, labels_male, 0.1)
        data_female, labels_female = select_Data_Subset(data_female, labels_female, 1)
        
        print "Shapes of Data (male, female) and sum of labels", np.shape(data_male), np.shape(data_female), sum(labels)
    
        data = np.concatenate((data_male, data_female))
        labels = np.concatenate((labels_male, labels_female))
        print "Data Subset Selected!"
    
        dataComplete = zip(data, labels)
        
        # SHUFFLE THE IMAGES
        random.shuffle(dataComplete)
        
        data = []
        labels = []
        for elem in dataComplete:
            data.append(elem[0])
            labels.append(elem[1])
        
        print "Saving the Data... (may take a while)"
        pickle.dump(data, open(pickle_directory + "data_vf.p", "wb"))        
        pickle.dump(labels, open(pickle_directory + "labels_vf.p", "wb"))        
        print "Data Saved, Good to Go!"
    
        print "Shapes of Data (male, female) and Labels", np.shape(data_male), np.shape(data_female), np.shape(labels)
        print "Sum of labels:", sum(labels)
        
    return data, labels, data

def getData_TestProtocol(source="train"):
    data_dir = "/Users/antonio/git/caltranscense/models/data/test_protocol/"
    pickle_directory = prm.params["pickle_directory"].get()
    
    data = []
    labels = []
    rawData = []
    
    try:
#         print "Loading the Data... (may take a while)"
        data = pickle.load(open(pickle_directory + "data_tp_" + str(source) + ".p", "rb"))
        labels = pickle.load(open(pickle_directory + "labels_tp_" + str(source) + ".p", "rb"))
#         print "Data Loaded, Good to Go!"
    except Exception as excp:  # data_vf.p doesnt exist
#         print "Exception:", excp
        for filename in os.listdir(data_dir):
            filename = filename.lower()
            if source in filename and filename[-3:] == "wav" and "," not in filename and "audio" not in filename and str(prm.params["device"].get()) in filename:
#                 if source == "train":
#                     if "clean" not in filename:
#                         continue
#                 if source == "test":
#                     if "clean" not in filename:
#                         continue
#                 print filename
        
                #######################################
                # GET RAW SIGNAL 
                #######################################
                params, rawSignal = get_RawSignal(data_dir + filename)
                prm.params["sample_rate"].set(params[2])
                            
                #######################################
                # CLEAN SAMPLES 
                #######################################
#                 fs, cleanSignal = remove_silence(params[2], rawSignal)
                cleanSignal = aeDetect(rawSignal)
                
                #######################################
                # FRAME SAMPLES 
                #######################################
                framedRawSignal = frame_chunks(cleanSignal)

                
                #######################################
                # ASCERTAIN GENDER 
                #######################################
                male_list = ["nigil", "leonard", "antonio", "thibault"]
                
                label = 0
                for male in male_list:
                    if male in filename:
                        label = 1
                        break
                
                for chunk in framedRawSignal:
                    feature_list = extractFeatures(chunk)
                    if feature_list != []:                          
                        data.append(feature_list)
                        labels.append(label)
                        rawData.append(chunk)
    #                         print np.shape(data), np.shape(data[-1]), np.shape(labels)
        
        print "Shapes of Data, Labels, RawData", np.shape(data), np.shape(labels), np.shape(rawData)
        data_flat = []
        data = np.array(data)
        for elem in data:
            data_flat.append(elem.flatten())
        data = data_flat
#         print "Shapes of Data and Labels", np.shape(data), np.shape(labels)


        #######################################
        # SHUFFLE THE IMAGES
        #######################################
        dataComplete = zip(data, labels, rawData)
        
        random.shuffle(dataComplete)
        
        data = []
        labels = []
        rawData = []
        for elem in dataComplete:
            data.append(elem[0])
            labels.append(elem[1])
            rawData.append(elem[2])

        
#         print "Saving the Data... (may take a while)"
        pickle.dump(data, open(pickle_directory + "data_tp_" + str(source) + ".p", "wb"))        
        pickle.dump(labels, open(pickle_directory + "labels_tp_" + str(source) + ".p", "wb"))        
#         print "Data Saved, Good to Go!"

#     print "Shapes of Data and Labels", np.shape(data), np.shape(labels), np.shape(rawData)
#     print "Sum of labels:", sum(labels)

    return data, labels, rawData


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
#     print "Actual   :", len(labels[0]), sum(labels[0])             
    for i in range(folds):
        # Initialize variables
        clf_local = clf
        j = 0
        accuracy = 0

        try:
            clf_local.fit(data[i], labels[i])
            # For each validation performed (k-1 total) on a fold
            for j in range(folds):
                if(j != i):
                    predicted_Labels = clf_local.predict(data[j])
                    for (elem1, elem2) in zip(predicted_Labels, labels[j]):
                        if elem1 == elem2:
                            accuracy += 1
                        else:
                            pass
                j += 1
#             print "Predicted:", len(predicted_Labels), sum(predicted_Labels)
            scores.append(100.0 * accuracy / ((folds - 1) * len(predicted_Labels)))
            i += 1
        except Exception, e:
            pass
#             print str(e)
    return np.array(scores)
                
def cleanPickle():
    pickle_dir = prm.params["pickle_directory"].get()

    for filename in os.listdir(pickle_dir):
        if "clf" not in filename:
            os.remove(pickle_dir + filename)
    
    for filename in os.listdir("./Errors/"):
        os.remove("./Errors/" + filename)
