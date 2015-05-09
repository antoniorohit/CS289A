** Real-Time Gender Classifier Based on Voice **

** Introduction **

This project documents a real-time gender classifier based on voice, developed as the final project for CS289A Introduction to Machine Learning (Spring 2015) at UC Berkeley.

The project uses a Random Forest classifier and MFCC+pitch features. A simplistic adaptive energy detection method is used to remove silence. It achieves >90% accuracy on a custom dataset, and has been tested on the voxforge dataset with good results as well. For more details on the implementation and tests, please read the report *Real Time Gender Identifier Based on Voice.pdf* 

** How to use the code **

**Dataset**

Before being able to run the classifier, you need to download the datasets. The Test_Protocol dataset is found here:
https://www.dropbox.com/sh/yfwi4kmt2jppt85/AABIVLAmDFjBcOLhiZmP-7cBa?dl=0

and these files should be placed in a ./Data/Test_Protocol/ folder in the directory structure. 

The dataset of voxforge files from:
http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/

should be placed in the ./Data/Voxforge/ folder.

** Important Files **
- main.py
- build.py
- brewed.py
- realTimeTest.py
