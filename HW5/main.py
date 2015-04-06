from Decision_Tree import DTree
import numpy as np
from scipy import io

############# FILE STUFF ############# 
File_Spam = "./spam_data.mat"
reg = 0.001
step_size = 0.000001
NUM_ITER = 1000

trainMatrix = io.loadmat(File_Spam)                 # Dictionary

############# GET DATA ############# 
training_data_raw = np.array(trainMatrix['training_data'])
trainingLabels_raw = np.array(trainMatrix['training_labels'])
testData = np.array(trainMatrix['test_data'])


