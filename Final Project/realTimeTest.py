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
RECORD_SECONDS = 20.

import os
from build import build

prm.params["chunk_size"].set(1)
build()

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

clf = pickle.load(open("./Pickle/clf_" + str(1.0) + ".p", "rb"))
    

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=False,
                frames_per_buffer=chunk)

print "* recording"

frames = []
for i in range(0, int(44100. / chunk * RECORD_SECONDS)):
    data = stream.read(chunk)
    frames.append(np.fromstring(data, "int16"))
    if((i+1)%4 == 0):
        data = np.array(frames[-4:]).ravel()
        _, data = remove_silence(44100, data)
        if len(data) > 0.25*44100:
            data = frame_chunks(data)[0]
            features = np.array(extractFeatures(data)).flatten()
            print i, "Female" if clf.predict(features.flatten()) == 0 else "Male"
        else:
            print i, "Silence"

print "* done"

stream.stop_stream()
stream.close()
p.terminate()

print "Num frames:", len(frames)
frames = np.ravel(frames)
print np.shape(frames)


fs, data = remove_silence(44100, (frames))
framedSignal = np.array(frame_chunks(data))
print "Shape of framedSignal", np.shape(framedSignal)

for chunk in framedSignal:
    features = np.array(extractFeatures(chunk)).flatten()
    predictedLabel.append(clf.predict(features.flatten()))
    
print "Percentage Male:", float(sum(predictedLabel)*1.0/len(predictedLabel))*100, "%"


wf = wave.open('pyaudiotest.wav', 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes((frames))
wf.close()
