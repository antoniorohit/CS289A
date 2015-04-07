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

    
    # if left == 
    def add_Node(self):
        self.node_list.append(self.Node())
        self.node_index+=1
        return self.node_index
            
    def train(self, data, labels):
        self.size += 1
        print "Node Count:", self.size
        # While the tree is not at its max depth and the 
        # node is not pure, keep splitting it
        while self.size < self.depth and (sum(labels) < len(labels)) and len(data) > 0:
            
            split_feat, thresh = self.segmentor(data, labels, self.impurity)
            self.node_list[self.node_index].split_rule = (split_feat, thresh)
            curr_node = self.node_index    
            if int(thresh) > 0:
                multi_val_arr = [(x,l) for (x,l) in zip(data, labels) if x[split_feat] < thresh]
                self.add_Node()
                self.node_list[curr_node].left = self.node_index
                left_data = [row[0] for row in multi_val_arr]
                left_labels = [row[1] for row in multi_val_arr]            
                self.train(left_data, left_labels)
            
            multi_val_arr = [(x,l) for (x,l) in zip(data, labels) if x[split_feat] >= thresh]
            self.add_Node()
            self.node_list[curr_node].right = self.node_index
            right_data = [row[0] for row in multi_val_arr]
            right_labels = [row[1] for row in multi_val_arr]
            self.train(right_data, right_labels)
            print "Did right too!"
        
        return
    
    def predict(self, test_data):
        
        return