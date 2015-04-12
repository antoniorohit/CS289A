from Decision_Tree import DTree
import numpy as np
from sklearn import tree, ensemble
from main import segmentor, entropy_impurity, gini_impurity, load_data, computeCV_Score

def create_forest_data(trainingData, trainingLabels):
    # It works - I checked
    trainingComplete = np.array(zip(trainingData, trainingLabels))
    data_len = len(trainingData)
    indices = np.random.randint(data_len,size=data_len)
    trainingComplete_replaced = trainingComplete[indices,:]
    trainingData = trainingComplete_replaced[:,0].tolist()
    trainingLabels = trainingComplete_replaced[:,1].tolist()
    return trainingData, trainingLabels

def create_forest(clf, trainingData, trainingLabels, NUM_TREES=25):
    classifier_list = []
    for _ in range(NUM_TREES):
        forestData, forestLabels = create_forest_data(trainingData, trainingLabels)
        clf_new = clf
        clf_new.fit(forestData, forestLabels, RNO="sqrt")
        classifier_list.append(clf_new)
    return classifier_list

def predict_forest(classifier_list, testDatum):
    label = 0.
    for clf in classifier_list:
        label += int(clf.predict([testDatum,])[0])
    if label > 0 and label < 30:
        print label, len(classifier_list)
    return np.round(label/len(classifier_list))

def computeCV_Score_Forest(clf, data, labels, folds, NUM_TREES=25):
    i = 0
    j = 0
    accuracy = 0.0
    scores = []
    # For each fold trained on...
    for i in range(folds):
        # Initialize variables
        j = 0
        accuracy = 0
        classifier_list = []
        for _ in range(NUM_TREES):
            forestData, forestLabels = create_forest_data(data[i], labels[i])
            clf_new = clf
            clf_new.fit(forestData, forestLabels, RNO="sqrt")
            classifier_list.append(clf_new)
        # For each validation performed (k-1 total) on a fold
        for j in range(folds):
            if(j!=i):
                predicted_Class = []
                for elem in data[j]:
                    predicted_Class.append(predict_forest(classifier_list, elem))
                for (elem1, elem2) in zip(predicted_Class, labels[j]):
                    if elem1 == elem2:
                        accuracy+=1                
            j+=1
        scores.append(100.0*accuracy/((folds-1)*len(predicted_Class)))
        i+=1
    return np.array(scores)

if __name__ == "__main__":
    print 50*'*'
    print "RANDOM FORESTS"
    print 50*'*'
    DEPTH = 25
    NUM_TREES = 50
    depths = [5, 25, 50]
    
    ############# FILE STUFF ############# 
    File_Spam = "./Data/spam_data.mat"
    
    trainingData, trainingLabels, testData = load_data(File_Spam)
    print "Num Features", np.shape(trainingData)[1]
    ############# DATA PARTIONING ############# 
    crossValidation_Data= []
    crossValidation_Labels = []
    k = 10 
    stepLength = k
    for index in range(0,k):
        crossValidation_Data.append(trainingData[index:-1:stepLength])
        crossValidation_Labels.append(trainingLabels[index:-1:stepLength])
    
    scoreBuffer = []
    
    ############# CROSS-VALIDATION ############# 
    print 50*'='
    print "CROSS VALIDATION USING RANDOM FORESTS"
    print 50*'='
    
    print "Num Trees:", NUM_TREES
    print 50*'-'

    for depth in depths:
        print "DEPTH:", depth
        clf = DTree(depth, gini_impurity, segmentor)
        scores = computeCV_Score_Forest(clf, crossValidation_Data, crossValidation_Labels, k, NUM_TREES)
        scoreBuffer.append((scores).mean())
        print "Depth:", depth, "Accuracy: %0.2f%% (+/- %0.2f)" % ((scores).mean(), np.array(scores).std() / 2)
        print 50*'-'
    
    maxScore = np.max(scoreBuffer)
    maxScore_Index = scoreBuffer.index(maxScore)
    print "Best Depth Value:", depths[maxScore_Index], "Accuracy for that Depth:", np.around(maxScore,3)
    print 50*'-'
    
    print "Creating best ever random forest based on Cross Validation!"
    clf_best = DTree(depths[maxScore_Index], gini_impurity, segmentor)
    best_forest = create_forest(clf_best, trainingData, trainingLabels, NUM_TREES)
    
    ############# TESTDATA PREDICT! ############# 
    predictedClass = []
    for elem in testData:
        predictedClass.append(predict_forest(best_forest, elem))

    print "Used this to predict classes on Test Data!"
    print 50*'-'
    
    ############# FOR KAGGLE ############# 
    indices = np.array(range(1,len(testData)+1))
    kaggle_format =  np.vstack(((indices), predictedClass)).T
    np.savetxt("./Results/spam.csv", kaggle_format, delimiter=",", fmt = '%d,%d',   header = 'Id,Category', comments='') 

    ############# BUILT-IN FUNCTION ############# 
    scoreBuffer = []
    print 50*'='
    print "CROSS VALIDATION USING SCIKIT-LEARN"
    print 50*'='
    
    for depth in depths:
        print "DEPTH:", depth
        clf = ensemble.RandomForestClassifier(n_estimators=5, criterion='entropy', max_depth=depth)    
        scores = computeCV_Score(clf, crossValidation_Data, crossValidation_Labels, k)
        scoreBuffer.append((scores).mean())
        print "Depth:", depth, "Accuracy: %0.2f%% (+/- %0.2f)" % ((scores).mean(), np.array(scores).std() / 2)
        print 50*'-'
    
    maxScore = np.max(scoreBuffer)
    maxScore_Index = scoreBuffer.index(maxScore)
    print "Best Depth Value:", depths[maxScore_Index], "Accuracy for that Depth:", np.around(maxScore,3)
    print 50*'-'
    
    print 20*"*", "The End" ,20*"*"
