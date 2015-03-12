import numpy as np
from numpy import linalg
import os
import cPickle as pickle
import pylab as plt

TRAIN_COUNT = 463715
TEST_COUNT = 51630

def GetMusicData(filename):
    f = open(filename, 'rb')
    music_data = []
    labels = []
    for line in f:
        temp_list = line.split(',')
        music_data.append(temp_list[1:])
        labels.append(temp_list[0])
    return music_data, labels

def ImportMusicData(filename):
    music_data = []
    labels = []
    if(os.path.isfile("./Results/labels.p")):
        print('opening data with pickle...')
        music_data = pickle.load(open("./Results/music_data.p", 'rb'))
        print('data successfully opened')
        print('opening labels with pickle..')
        labels = pickle.load(open("./Results/labels.p", 'rb'))
        print('labels successfully opened')
    else:
        print "PICKLE FILE NOT FOUND"
        music_data, labels = GetMusicData(filename)
#         pickle.dump(music_data, open("./Results/music_data.p", 'wb'))
#         pickle.dump(labels, open("./Results/labels.p", 'wb'))
        print "Pickle Files Saved!"
    return np.array(music_data, dtype = float), np.array(labels, dtype = float)

def TrainLeastSquares(train_data, train_labels):
    train_data = np.matrix(train_data)
    train_labels = np.matrix(train_labels).T
    A = np.transpose(train_data)*train_data
    b = train_data.T*train_labels
    beta = linalg.solve(A,b)
    return beta

filename = "YearPredictionMSD.txt"
music_data, labels =  ImportMusicData(filename)

# Data augmentation for the bias term
music_data = np.hstack((np.ones((TRAIN_COUNT+TEST_COUNT, 1)), music_data))

print "Data shape:", music_data.shape, "labels shape:", labels.shape

train_data = music_data[:TRAIN_COUNT]
train_labels = labels[:TRAIN_COUNT]

test_data = music_data[TRAIN_COUNT:]
test_labels = labels[TRAIN_COUNT:]

beta = TrainLeastSquares(train_data, train_labels)

print "Beta Shape:", beta.shape

y = np.around(np.matrix(test_data)*beta, 2)

print "Min Predicted:", min(y), "Max Predicted:", max(y)

plt.close()
plt.stem(beta[1:])
plt.savefig("./Results/BetaPlot.png")

RSS = 0.0
accuracy = 0.0
for elem1, elem2 in zip(y, test_labels):
    if elem1 == elem2:
        accuracy+=1
    else:
        pass
    
    RSS += (elem1 - elem2)**2

print RSS