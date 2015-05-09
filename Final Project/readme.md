**Real-Time Gender Classifier Based on Voice**

**Introduction**

This project documents a real-time gender classifier based on voice, developed as the final project for CS289A Introduction to Machine Learning (Spring 2015) at UC Berkeley.

The project uses a Random Forest classifier and MFCC+pitch features. A simplistic adaptive energy detection method is used to remove silence. It achieves >90% accuracy on a custom dataset, and has been tested on the voxforge dataset with good results as well. For more details on the implementation and tests, please read the report *Real Time Gender Identifier Based on Voice.pdf* 

**How to use the code**

**Dataset**

Before being able to run the classifier, you need to download the datasets. The Test_Protocol dataset is found here:
https://www.dropbox.com/sh/yfwi4kmt2jppt85/AABIVLAmDFjBcOLhiZmP-7cBa?dl=0

and these files should be placed in a ./Data/Test_Protocol/ folder in the directory structure. 

The dataset of voxforge files from:
http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/

should be placed in the ./Data/Voxforge/ folder.

**Important Files**
- main.py

Running main allows you to build your data, train the classifier, and test it out. 
The build function builds a classifier based on the parameters you choose based on cross-validation (see build.py and train.py). The train function contains code that cross-validates your classifier based on depth. The test function uses the test set to evaluate accuracy of the classifier.

- realTimeTest.py

After a classifier is built by running main.py, you can evaluate its real-time performance by running realTimeTest and speaking into your microphone. This file prints to the console whether the speaker is male, female or if silence is detected.

- ./pitch_extraction/simple_pitch.py

This file if run directly, reads in a wav file specified by you, and plays it back chunk by chunk, estimating the pitch for each chunk, and also specifying gender based on this pitch

- ./feature_extaction/extract_features.py

This file provides an easy way to tweak features fed into the classifier

- ./VAD/aed.py 

Adaptive energy detection method for signal cleaning

- ./Errors/

Misclassified chunks are placed into this folder

