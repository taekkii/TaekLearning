
import os
import sys
import importlib
import torch.nn as nn

from .model import get

EXCPETIONAL_DIRECTORIES = ['__pycache__']


def get_all_model_names()->list:
    
    file_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(file_path)
    return [entry.name for entry in os.scandir(dir_path) if entry.is_dir()  and entry.name not in EXCPETIONAL_DIRECTORIES ]



def get_model_dict(lowercase=False)->dict:

    model_dict = {}
    for model_name in get_all_model_names():
        imported_module_dict = importlib.import_module('.'+model_name,__name__).__dict__
    
        for attriute_name,attribute in imported_module_dict.items():
            if type(attribute)==type  and  issubclass(attribute,nn.Module):
                if lowercase: model_dict[model_name.lower()]=attribute
                else        : model_dict[model_name        ]=attribute
    
    return model_dict

for model_name, model_class in get_model_dict().items():
    setattr( sys.modules[__name__], model_name , model_class )
    