'''
Created on Apr 21, 2015

@author: antonio
'''
import pyaudio
import sys
import parameters as prm

sys.path.append("./VAD")
sys.path.append("./feature_extraction")


import wave 
from _imaging import path
from simple import remove_silence
from extract_features import extractFeatures
from utils import *
import cPickle as pickle

chunk = 11025
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 10.

import os
     

p = pyaudio.PyAudio()

if 0:
    try:
        user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
    except KeyError:
        user_paths = []
    for path in sys.path:
        print path
    print p.get_device_info_by_index(0)['defaultSampleRate']

predictedLabel = []

clf = pickle.load(open("./Pickle/clf_" + str(prm.params["chunk_size"].get()) + ".p", "rb"))
    

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=False,
                frames_per_buffer=chunk)

frames = []

print "* recording"

for i in range(0, int(44100. / chunk * RECORD_SECONDS)):
    print i
    data = stream.read(chunk)
    frames.append(np.fromstring(data, "int16"))
    # check for silence here by comparing the level with 0 (or some threshold) for 
    # the contents of data.
    # then write data or not to a file

print "* done"

stream.stop_stream()
stream.close()
p.terminate()

# frames = np.fromstring(frames, "int16")
print "Num frames:", len(frames)
frames = np.ravel(frames)
print type(frames[0]), np.shape(frames)



fs, data = remove_silence(44100, (frames))
framedSignal = np.array(frame_chunks(data))
print "Shape of framedSignal", np.shape(framedSignal)

for chunk in framedSignal:
    features = np.array(extractFeatures(chunk)).flatten()
    print "Shape of features", np.shape(features)
    predictedLabel.append(clf.predict(features.flatten()))
    
print predictedLabel


wf = wave.open('pyaudiotest.wav', 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes((frames))
wf.close()
