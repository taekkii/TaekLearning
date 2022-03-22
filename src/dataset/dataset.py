


import os
from ..utils import config

EXCPETIONAL_DIRECTORIES = ['__pycache__']

class DatasetNotExistError(Exception):
    pass


def get_all_dataset_names()->list:
    
    file_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(file_path)
    return [entry.name for entry in os.scandir(dir_path) if entry.is_dir()  and entry.name not in EXCPETIONAL_DIRECTORIES ]





def get_dataset( dataset_name:str , dataset_config:dict ):
    
    dataset_dict = { k.lower():v for k,v in config.get_dataset_dict().items() }
    
    #----- GUARD -----#
    if dataset_name.lower() not in dataset_dict:
        raise DatasetNotExistError(f"Unregistered dataset : [{dataset_name}]")
    
    dataset_name = dataset_name.lower()

    return dataset_dict[dataset_name](**dataset_config)


