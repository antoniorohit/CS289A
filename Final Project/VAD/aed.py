'''
Created on Apr 29, 2015

@author: antonio
'''
import numpy as np
import scipy as sp
import parameters as prm

# @timing
def aeDetect(raw_signal):
    """ Adaptive energy detection
    see:        http://homepage.tudelft.nl/w5p50/pdffiles/VAD%20Techniques%20for%20Real-Time%20Speech%20Transmission%20on%20the%20Internet.pdf 
    :return: cleaned signal
    """
#     offset = prm.params["offset"].get()
    offset = np.argmin(abs(raw_signal))
    region_size = 440
    var = np.var(raw_signal)
    
    if((offset+region_size/2. < len(raw_signal)) and (offset-region_size/2. > 0)):
        SIGNAL_THRESH = 1.5*np.average(abs(raw_signal[offset-region_size/2.:offset + region_size/2.]))
#         print "Thresh: ", SIGNAL_THRESH, offset  
#         print "Var:    ", np.var(raw_signal) 
#         print "Len:    ", len(raw_signal)     
    else:
        SIGNAL_THRESH = 100
  
    p = 0.1
    old_var = 1.0
        
    # initialize a new array of 16bit integers - the wav file is 16bit
    cleaned_signal = sp.zeros(len(raw_signal), sp.int16)
                
    # Go through the wav file in chunks of aed_params["region_size"].get()
    # check if the region is a 'zero region', and if so, clear the signal to 0
    # in that region
    
    half_region = int(region_size/2.)
    rsLen = len(raw_signal)
        
    # start i from aed_params["region_size"].get()/2 so that we dont under or overflow array limits    
    i = half_region
    region=2*half_region
        
    while i <= (rsLen - half_region):
        i1 = i - half_region
        i2 = i + half_region
        signal_of_interest = (raw_signal[i1:i2])
        average_local = np.average(abs(signal_of_interest))
        if average_local < SIGNAL_THRESH:
            SIGNAL_THRESH = (1-p)*SIGNAL_THRESH + 1.5*p*average_local
            cleaned_signal[i1:i2] = sp.zeros(region, sp.int16)
            new_var = np.var(signal_of_interest)
            if(new_var/old_var > 1.25):
                p = 0.25
            else:
                p = 0.1
    
            old_var = new_var
        else:
            cleaned_signal[i1:i2] = signal_of_interest
        i += region
    
    # get rid of zero regions
    cleaned_signal = cleaned_signal[np.nonzero(cleaned_signal)]
              
    if len(cleaned_signal) * 2 < len(raw_signal):
        return []
    
    return raw_signal