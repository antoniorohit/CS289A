import numpy as np

class DTree(object):
    """ Decision Tree Object """
    
    class Node(object):
        """Definition of a regular Node"""
        
        def __init__(self, split_rule=None, left=None, right=None):
            self.left = left
            self.right = right
            self.split_rule = split_rule
            return
    
    class Leaf_Node(object):
        """Definition of a Leaf Node"""
        
        def __init__(self, label):
            self.label = label         
    
    def __init__(self, depth, impurity, segmentor):
        """Return a Dtree object"""
        self.depth = depth
        self.impurity = impurity
        self.segmentor = segmentor
        self.size = 0
        self.root = 0
        self.node_list = []
        self.node_index = -1
        # ROOT
        self.add_Node()
    
    def __len__(self):
        return self.size

    
    def add_Node(self):
        self.node_list.append(self.Node())
        self.node_index+=1
        return self.node_index
            
    def train(self, train_data, train_labels):
        local_data = train_data
        local_labels = train_labels
        
        # While the tree is not at its max depth and the 
        # node is not pure, keep splitting it
        while self.size < self.depth and (sum(local_labels) < len(local_labels)):
            split_feat, thresh = self.segmentor(local_data, local_labels, self.impurity)
            self.node_list[self.node_index].split_rule = (split_feat, thresh)
            local_data = [x for x in local_data if x[split_feat] < thresh]
            
            pass
        
        return
    
    def predict(self, test_data):
        return