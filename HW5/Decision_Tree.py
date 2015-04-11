import time

class DTree(object):
    """ Decision Tree Object """
    
    class Node(object):
        """Definition of a regular Node"""
        
        def __init__(self, split_rule=None, left=None, right=None):
            self.left = left
            self.right = right
            self.split_rule = split_rule
            self.type = "Node"
            return
        
        def __type__(self):
            return "Node"
    
    class Leaf_Node(object):
        """Definition of a Leaf Node"""
        
        def __init__(self, label):
            self.type = "Leaf"
            self.label = label         
    
    def __init__(self, depth, impurity, segmentor):
        """Return a Dtree object"""
        self.max_depth = depth
        self.impurity = impurity
        self.segmentor = segmentor
        self.depth = 0
        self.root = 0
        self.node_list = []
        self.node_index = -1
        self.VISUALIZE = False      # visualize splits on prediction
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
        return self.node_index
    
    def fit(self, data, labels):
#         print self.depth
        curr_node = self.node_index    
        split_feat, thresh = self.segmentor(data, labels, self.impurity)
        self.node_list[self.node_index].split_rule = (split_feat, thresh)
        # While the tree is not at its max depth and the 
        # node is not pure, keep splitting it
        if self.depth < self.max_depth and thresh > 0:
#             self.size_l += 1
            multi_val_arr = [(x,l) for (x,l) in zip(data, labels) if x[split_feat] < thresh]
            left_data = [row[0] for row in multi_val_arr]
            left_labels = [row[1] for row in multi_val_arr]
            if(sum(left_labels) == 0):
                self.add_LNode(0)
                self.node_list[curr_node].left = self.node_index
            elif(sum(left_labels) == len(left_labels)):          
                self.add_LNode(1)
                self.node_list[curr_node].left = self.node_index
            else:
                self.depth += 1
                self.add_Node()
                self.node_list[curr_node].left = self.node_index
                self.fit(left_data, left_labels)            
                # exiting one node
                self.depth -=1
        
#             self.size_r += 1
            multi_val_arr = [(x,l) for (x,l) in zip(data, labels) if x[split_feat] >= thresh]
            right_data = [row[0] for row in multi_val_arr]
            right_labels = [row[1] for row in multi_val_arr]
            if(sum(right_labels) == 0):
                self.add_LNode(0)
                self.node_list[curr_node].right = self.node_index
            elif(sum(right_labels) == len(right_labels)):          
                self.add_LNode(1)
                self.node_list[curr_node].right = self.node_index
            else:
                self.depth += 1
                self.add_Node()
                self.node_list[curr_node].right = self.node_index
                self.fit(right_data, right_labels)
                # exiting one node
                self.depth -=1
                        
        else:  # Thresh is zero or  depth > max_depth
            if(sum(labels)/len(data) < 0.5):
                self.add_LNode(0)
            else:
                self.add_LNode(1)
            # no split is possible - the node is a leaf, so replace it
            self.node_list[curr_node] = self.node_list.pop()
            self.node_index-=1      # fix the node index 
        
        return
    
    def predict(self, test_data):
        predictedLabels = []
        loc_node = self.Node()
        for elem in test_data:
            loc_node = self.node_list[0]            # ROOT
            while (loc_node.type == "Node"):
#                 print loc_node.left, loc_node.right, loc_node.split_rule
                split_feat, thresh = loc_node.split_rule
                if self.VISUALIZE == True:
                    print split_feat, thresh 
                    time.sleep(.1)    # pause 
                if(elem[split_feat] < thresh):
#                     print "L"
                    loc_node = self.node_list[loc_node.left]
                else:
#                     print "R" 
                    loc_node = self.node_list[loc_node.right]
            predictedLabels.append(loc_node.label)

        return predictedLabels