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
        self.size = 0
        self.root = 0
    
    def __len__(self):
        return self.size

    def train(self, train_data, train_labels):
        local_data = train_data
        local_labels = train_labels
        
        # While the tree is not at its max depth and the 
        # node is not pure, keep splitting it
        while self.size < self.depth and (sum(local_labels) < len(local_labels)):
            split_feat, thresh = self.segmentor(local_data, local_labels, self.impurity)
            
            pass
        
        return
    
    def predict(self, test_data):
        return