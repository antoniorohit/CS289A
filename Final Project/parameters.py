'''
Created on Apr 13, 2015

@author: antonio
'''

class Parameter:
    '''class allowing to store both default and all possible values for each parameter'''
    
    def __init__(self, default_value, all_values=list(), actions=dict()):
        
        # default value initialized with argument
        self.default_value = default_value
        
        # possible values initialized with argument
        self.values = all_values
        
        # current value initialized to default value
        self.current_value = default_value
        
        # name initialized to empty string
        self.name = str()
        
        # actions initialized to argument
        self.actions=actions
    
    def default(self):
        ''' get default value.
        
        :return: self.default_value: default value'''
        return self.default_value
    
    def all(self):
        '''get all possible values.
        
        :return: self.values : all possible values'''
        return self.values
    
    def get(self):
        '''set and return current new_value.
        :return: current value'''
        
        # return current value
        return self.current_value
        
    def set(self, new_value=None):
        '''set and return current new_value.        
        :param new_value : new value to set current value
        :return: current value'''
        
        # if parameter current value is already set to new_value tell user and return current value
        if new_value==self.current_value:
            print(self.name + " parameter set to " + str(new_value) + " (default value)")
            return self.current_value
        
        # set current value to new value and tell user
        self.current_value = new_value
        print(self.name + " parameter set to " + str(new_value))
        
        # return current value
        return self.current_value

    def launch(self, *args, **kwargs):
        """return result of action associated with current value.
        
        :param any useful argument for the action to be performed
        :return: result of the action associated with the current value"""
        
        # if no actions has been defined tell so user and do nothing
        if len(self.actions.keys())==0:
            print(self.name + " parameter has not been associated with any action, launch impossible")
            return
        
        # else launch action and return function
        return self.actions[self.current_value](*args, **kwargs)
        
    
# declare params

global params
    
# initialize dictionary
params=dict()

# fill dictionary with values
params["sample_rate"] = Parameter(16000, [8000, 16000, 44100])            # hz
params["chunk_size"] = Parameter(1, [0.250, 0.500, 1.000, 2.000])              # s
params["data_directory"] = Parameter("./Data/")

# set names
for key in params.keys():
    params[key].name=key   