

import utils.config
from torch.utils.data import Dataset
#from . import get_dataset_dict

class DatasetNotExistError(Exception):
    pass






def get( dataset_name:str , dataset_config:dict )->Dataset:
    
    dataset_dict = {k.lower():v for k,v in get_dataset_dict().items()}
    
    #----- GUARD -----#
    if dataset_name.lower() not in dataset_dict:
        raise DatasetNotExistError(f"Unregistered dataset : [{dataset_name}]")
    
    dataset_name = dataset_name.lower()

    return dataset_dict[dataset_name](**dataset_config)


