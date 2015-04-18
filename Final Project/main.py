'''
Created on Apr 13, 2015

High-Level Code which is called to test or train the system

@author: antonio
'''
from train import train
from test import test
import parameters as prm

prm.params["chunk_size"].set(0.25)

train()

test()

