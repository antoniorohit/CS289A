import numpy as np

class DTree(object):
    """ Decision Tree Object """
    
    class Node(object):
        """Definition of a regular Node"""
        
        def __init__(self, split_rule, left, right):
            return
    
    class Leaf_Node(object):
        """Definition of a Leaf Node"""
        
        def __init__(self, label):
            return        
    
    def __init__(self, depth, impurity, segmentor):
        """Return a Dtree object"""
        self.depth = depth
        self.impurity = impurity
        self.segmentor = segmentor
    
    def train(self, train_data, train_labels):
        return
    
    def predict(self, test_data):
        return