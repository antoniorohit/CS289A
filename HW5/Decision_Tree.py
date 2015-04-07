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
            
    def add_LNode(self, label):
        self.node_list.append(self.Leaf_Node(label))
        self.node_index+=1
        print "LEAF!"
        return self.node_index
    
    def train(self, data, labels):
        self.size += 1
        print "Node Count:", self.size
        # While the tree is not at its max depth and the 
        # node is not pure, keep splitting it
        if (self.size < self.depth and (sum(labels) != len(labels)) and (sum(labels) > 0) and len(data) > 0):
            split_feat, thresh = self.segmentor(data, labels, self.impurity)
            self.node_list[self.node_index].split_rule = (split_feat, thresh)
            curr_node = self.node_index    
            if (thresh) > 0:
                print thresh
                multi_val_arr = [(x,l) for (x,l) in zip(data, labels) if x[split_feat] < thresh]
                left_data = [row[0] for row in multi_val_arr]
                left_labels = [row[1] for row in multi_val_arr]
                if(sum(left_labels) == 0):
                    self.add_LNode(0)
                elif(sum(left_labels) == len(left_labels)):          
                    self.add_LNode(1)
                else:
                    self.add_Node()
                    self.node_list[curr_node].left = self.node_index
                    print "LEFT"
                    self.train(left_data, left_labels)
            
                multi_val_arr = [(x,l) for (x,l) in zip(data, labels) if x[split_feat] >= thresh]
                right_data = [row[0] for row in multi_val_arr]
                right_labels = [row[1] for row in multi_val_arr]
                if(sum(right_labels) == 0):
                    self.add_LNode(0)
                elif(sum(right_labels) == len(right_labels)):          
                    self.add_LNode(1)
                else:
                    self.add_Node()
                    self.node_list[curr_node].right = self.node_index
                    print "RIGHT"
                    self.train(right_data, right_labels)
            
        
        return
    
    def predict(self, test_data):
        
        return