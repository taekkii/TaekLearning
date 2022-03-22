

import os



#from . import get_model_dict



def get( model_name:str , model_config:dict ):

    model_dict = get_model_dict(lowercase=True)


    #----- GUARD -----#
    assert model_name.lower() in model_dict , f"Unregistered model : [{model_name}]"
    
    dataset_name = dataset_name.lower()

    return model_dict[dataset_name](**model_config)
    
