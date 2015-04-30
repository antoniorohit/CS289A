# Read in a WAV and find the freq's
import sys

sys.path.append("../VAD")

import pyaudio
import wave
import numpy as np
from simple import remove_silence
from aed import aeDetect
chunk = 8820

# open up a wave
wf = wave.open("/Users/antonio/git/caltranscense/models/data/test_protocol/android_helene_train_clean_21.wav", 'rb')
wf = wave.open("/Users/antonio/git/caltranscense/models/data/test_protocol/android_leonard_train_clean_36.wav", 'rb')

swidth = wf.getsampwidth()
RATE = wf.getframerate()
# use a Blackman window
window = np.hamming(chunk)
# open stream
p = pyaudio.PyAudio()
stream = p.open(format=
                p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=RATE,
                output=True)

female = 0
male = 0
silence = 0
# read some data
data = wf.readframes(chunk)
# play stream and find the frequency of each chunk
while len(data) == chunk * swidth:
    # write data out to the audio stream
    stream.write(data)
    # unpack the data and times by the hamming window
    indata = np.array(wave.struct.unpack("%dh" % (len(data) / swidth), \
                                         data)) * window
                                         
    indata = aeDetect(indata)
    if len(indata) > 0:
        # Take the fft and square each value
        fftData = abs(np.fft.rfft(indata)) ** 2
        # find the maximum
        which = fftData[10:60].argmax() + 10
        # use quadratic interpolation around the max
        if which != len(fftData) - 1:
            y0, y1, y2 = np.log(fftData[which - 1:which + 2:])
            x1 = (y2 - y0) * .5 / (2 * y1 - y2 - y0)
            # find the frequency and output it
            thefreq = (which + x1) * RATE / chunk
    #         print "The freq is %f Hz." % (thefreq)
        else:
            thefreq = which * RATE / chunk
    #         print "The freq is %f Hz." % (thefreq)
        if thefreq > 230:
            print "FEMALE", np.around(thefreq)
            female += 1
        else:
            print "MALE", np.around(thefreq)
            male += 1
        # read some more data
    else:
        print "SILENCE"
        silence += 1
    data = wf.readframes(chunk)
if data:
    stream.write(data)
stream.close()
p.terminate()

print male, female, silence, male+female+silence