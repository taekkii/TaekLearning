


import os
import sys
import importlib
from torch.utils.data import Dataset

from .dataset import get
from .transform import TRANSFORM_DICT

EXCPETIONAL_DIRECTORIES = ['__pycache__']

def get_all_dataset_names()->list:
    
    file_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(file_path)
    return [entry.name for entry in os.scandir(dir_path) if entry.is_dir()  and entry.name not in EXCPETIONAL_DIRECTORIES ]



def get_dataset_dict(lowercase=False)->dict:

    dataset_dict = {}
    for dataset_name in get_all_dataset_names():
        imported_module_dict = importlib.import_module('.'+dataset_name,__name__).__dict__
    
        for attriute_name,attribute in imported_module_dict.items():
            if type(attribute)==type  and  issubclass(attribute,Dataset):#
                if lowercase: dataset_dict[dataset_name.lower()] = attribute
                else        : dataset_dict[dataset_name        ] = attribute
    
    return dataset_dict


for dataset_name,dataset_class in get_dataset_dict().items():
    setattr( sys.modules[__name__], dataset_name , dataset_class )
    