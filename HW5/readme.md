How to use HW5_Antonio.zip
- Contains 2 high-level files: main.py and random_forest.py
- Running main.py executes code that trains and cross validates decision trees
- Running random_forest.py executes code that trains a random forest
- Result of the prediction on test data is put into spam.csv in the Results folder
- The Data folder contains spam_data.mat and featurize.py
- The decision tree class is implemented in Decision_Tree.py
- The random forest code is in random_forest.py and not yet encapsulated into a class
- main.py implements impurity, segmentor, load_data and compute_CV_score (cross-validation score) - useful functions re-used in random_forest.py
