

import sys
import os
import importlib
from .main import Task

EXCPETIONAL_DIRECTORIES = ['__pycache__']

def get_all_tasks_names()->list:
    
    file_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(file_path)
    return [entry.name for entry in os.scandir(dir_path) if entry.is_dir()  and entry.name not in EXCPETIONAL_DIRECTORIES ]


def get_task_dict(lowercase=False)->dict:

    task_dict = {}
 
    for task_name in get_all_tasks_names():
        imported_module_dict = importlib.import_module('.'+task_name,__name__).__dict__
    
        for attriute_name,attribute in imported_module_dict.items():
            if type(attribute)==type  and  issubclass(attribute,Task):
                if lowercase: task_dict[task_name.lower()] = attribute
                else        : task_dict[task_name        ] = attribute
    
    return task_dict


for task_name,task_class in get_task_dict().items():
    setattr(sys.modules[__name__], task_name , task_class)


