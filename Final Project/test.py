import sys

sys.path.append("./feature_extraction")

import os
import parameters as prm
from utils import *

data_directory = prm.params["data_directory"].get()
prm.params["chunk_size"].set(1)
            
data, labels = extract_Data(data_directory)


        

        