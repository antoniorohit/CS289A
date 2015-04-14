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
    
    # Zero padding for last sample (unnecessary?)    
    if index != len(rawSignal):
        framedRawSignal.append(rawSignal[index:])
        framedRawSignal[-1] = np.append(framedRawSignal[-1], np.zeros(chunk_size - len(framedRawSignal[-1])))

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
    labels = []
    folder = directory+"/etc/"
    readme =  open(folder+"README").read().strip().lower()
    if("male" in readme):
        labels.append(1)
    elif("female" in readme):
        labels.append(0)
    else:
        print "ERROR!!"
        print readme
        exit
    return labels

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
                        features = []
                        label = extract_Gender_Label(directory)
                        params, rawSignal = get_RawSignal(directory+folder+file_name)
                        framedRawSignal = (frame_chunks(rawSignal))
                        for chunk in framedRawSignal:                            
                            data.append(extractFeatures(chunk))
                            labels.append(label)
#                         print np.shape(data), np.shape(data[-1]), np.shape(labels)
                        
            except OSError:
                print "OS Error"
    return data, labels






