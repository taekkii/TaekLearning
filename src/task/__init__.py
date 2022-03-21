
#from .classification import ClassificationTask
from .task import get_all_tasks_names

import sys
import os
import importlib

for task_name in get_all_tasks_names():
    setattr(sys.modules[__name__], 
            task_name.capitalize()+'Task' , 
            getattr( importlib.import_module('.'+task_name,__name__) , task_name.capitalize()+'Task'  ) )




