# Read in a WAV and find the freq's

import pyaudio
import wave
import numpy as np
import parameters as prm
chunk = 44100

def extract_pitch(rawSignal):
    swidth = 2
    RATE = prm.params["sample_rate"].get()
    chunk = prm.params["chunk_size"].get()*RATE
    scale = RATE*1.0/(chunk)
    fftData = abs(np.fft.rfft(rawSignal)) ** 2
    # find the maximum
    lofreq = 50    # hz
    hifreq = 210    #hz
    which = fftData[lofreq/scale:hifreq/scale].argmax() + lofreq/scale
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
    
    return thefreq       


if __name__ == "__main__":
    import sys
    sys.path.append("../VAD")
    from simple import remove_silence
    from aed import aeDetect
    # open up a wave
    wf = wave.open("/Users/antonio/git/caltranscense/models/data/test_protocol/android_helene_train_clean_21.wav", 'rb')
#     wf = wave.open("/Users/antonio/git/caltranscense/models/data/test_protocol/android_leonard_train_clean_36.wav", 'rb')
#     wf = wave.open("/Users/antonio/git/caltranscense/models/data/test_protocol/Android_mathilde_test_BGN_42.wav", 'rb')
#     wf = wave.open("/Users/antonio/git/caltranscense/models/data/test_protocol/Android_antonio_test_clean_120.wav", 'rb')

    swidth = wf.getsampwidth()
    print swidth
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
    prev = ""
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
            thefreq = extract_pitch(indata)
            if thefreq > 160:
                if prev == "M":
                    print "MALE", np.around(thefreq)
                    male += 1
                    prev = "S"
                else:
                    print "FEMALE", np.around(thefreq)
                    female += 1
                    prev = "F"
    
            else:   # freq < 160
                if prev == "F":
                    print "FEMALE", np.around(thefreq)
                    female += 1
                    prev = "S"
                else:
                    print "MALE", np.around(thefreq)
                    male += 1
                    prev = "M"
            # read some more data
        else:
            if prev == "S":
                print "SILENCE"
                silence += 1
                prev = "S"
            else:
                if prev == "F":
                    print "FEMALE"
                    female += 1
                    prev = "S"
                else:
                    print "MALE"
                    male += 1
                    prev = "S"
                
                
        data = wf.readframes(chunk)
    if data:
        stream.write(data)
    stream.close()
    p.terminate()
    
    print male, female, silence, male+female+silence