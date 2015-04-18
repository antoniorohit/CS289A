'''
Created on Apr 13, 2015

High-Level Code which is called to test or train the system

@author: antonio
'''
from train import train
from test import test
from utils import cleanPickle
import parameters as prm
import os
    
    

prm.params["chunk_size"].set(0.25)

cleanPickle()

print 20*"#", "Training", 20*"#"
train()

print 20*"#", "Testing", 20*"#"
test()

