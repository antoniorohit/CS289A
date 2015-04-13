import numpy as np
import os
import sys
from collections import Counter
from collections import OrderedDict
import pylab as plt
import operator

ham_dir = "./Data/ham/"
spam_dir = "./Data/spam/"
test_dir = "./Data/test/"

ham_bag = []
spam_bag = []
test_bag = []

######################## HAM ########################
i = 0
NUM = 1000
for filename in os.listdir(ham_dir):
    i+=1
    if i > NUM:
        break
    if filename[-3:] == 'txt':
        f = open(ham_dir+filename, 'rb')
        x = [x for x in f.read().lower().split() if len(x) > 3]
        ham_bag = np.concatenate((ham_bag, x), 0)

cnt = Counter(ham_bag)
sorted_cnt = sorted(cnt.items(), key=operator.itemgetter(1), reverse=True)[0:30]
print 20*"*", "HAM", 20*"*"
print "Num words, files:", len(cnt), len(os.listdir(ham_dir))
print sorted_cnt

######################## SPAM ########################

i = 0
for filename in os.listdir(spam_dir):
    i+=1
    if i > NUM:
        break
    if filename[-3:] == 'txt':
        f = open(spam_dir+filename, 'rb')
        x = [x for x in f.read().lower().split() if len(x) > 3]
        spam_bag = np.concatenate((spam_bag, x), 0)


cnt = Counter(spam_bag)
sorted_cnt = sorted(cnt.items(), key=operator.itemgetter(1), reverse=True)[0:30]
print 20*"*", "SPAM", 20*"*"
print "Num words, files:", len(cnt), len(os.listdir(spam_dir))
print sorted_cnt

######################## TEST ########################

i = 0
for filename in os.listdir(test_dir):
    i+=1
    if i > NUM:
        break
    if filename[-3:] == 'txt':
        f = open(test_dir+filename, 'rb')
        x = [x for x in f.read().lower().split() if len(x) > 3]
        test_bag = np.concatenate((test_bag, x), 0)


cnt = Counter(test_bag)
sorted_cnt = sorted(cnt.items(), key=operator.itemgetter(1), reverse=True)[0:30]
print 20*"*", "TEST", 20*"*"
print "Num words, files:", len(cnt), len(os.listdir(test_dir))
print sorted_cnt

print 20*"*", "THE END", 20*"*"
