

import dataset
import model
from typing import Iterable
import inspect 
import dataset.transform

import utils.dictionarylike
import utils.predefine

def _get_filtered_dict(original_dict:dict , include:Iterable[str] ):    
    return{k:v for k,v in original_dict.items() if k in include}

def prepare_dataset(args:dict):

    result_dict = {}
    dictionarylike_statements = args.get('dataset',[])

    for dictionarylike_statement in dictionarylike_statements:
        parsed = utils.dictionarylike.parse(dictionarylike_statement)
        
        if len(parsed)==1 and '_key' in parsed:
            parsed['config']=parsed['_key']
        
        if 'config' in parsed:
            original_key = parsed['_key']
            config_key = parsed['config']
            parsed = utils.predefine.load(config_key,yaml_filename='dataset')
            parsed['_key'] = original_key

        if 'transform' in parsed:
            parsed['transform'] = dataset.TRANSFORM_DICT[parsed['transform']]
        
        key = parsed['_key']
        dataset_name = parsed['name']
        
        dataset_args = inspect.getfullargspec( dataset.get_dataset_dict(lowercase=True)[dataset_name.lower()].__init__ ).args
        dataset_config = _get_filtered_dict(parsed,dataset_args)

        result_dict[key] = dataset.get(dataset_name=dataset_name , dataset_config=dataset_config)

    args['dataset'] = result_dict



def prepare_model(args:dict):
    result_dict = {}
    dictionarylike_statements = args.get('model',[])
    
    for dictionarylike_statement in dictionarylike_statements:
        
        parsed = utils.dictionarylike.parse(dictionarylike_statement)

        if len(parsed)==1 and '_key' in parsed:
            parsed['config']=parsed['_key']

        if 'config' in parsed:
            original_key = parsed['_key']
            config_key = parsed['config']
            parsed = utils.predefine.load(config_key,yaml_filename='model')
            parsed['_key'] = original_key
            

        net_name = parsed['_key']
        model_name = parsed['model']
        model_args = inspect.getfullargspec( model.get_model_dict(lowercase=True)[model_name.lower()].__init__  ).args
        model_config = _get_filtered_dict(parsed,model_args)

        result_dict[net_name] = model.get(model_name=model_name , model_config=model_config)

    args['model'] = result_dict
            
    

def prepare_trainchunk(args:dict):
    result_list = []
    dictionarylike_statements = args.get('trainchunk',[])

    for dictionarylike_statement in dictionarylike_statements:
        
        parsed = utils.dictionarylike.parse(dictionarylike_statement)

        if 'config' in parsed:
            pass

        result_list.append(parsed)

    args['trainchunk'] = result_list