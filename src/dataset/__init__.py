

from .dataset import get_all_dataset_names
from .dataset import get_dataset

import sys
import importlib

for dataset_name in get_all_dataset_names():
    setattr(sys.modules[__name__], 
            dataset_name+'Dataset' , 
            getattr( importlib.import_module('.'+dataset_name,__name__) , dataset_name+'Dataset'  ) )



