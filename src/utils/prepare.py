

import dataset
import model
from typing import Iterable
import inspect 
import dataset.transform

import utils.dictionarylike
import utils.predefine

import torch

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
            config_key = parsed['config']
            loaded = utils.predefine.load(key=config_key , yaml_filename='dataset')
            
            loaded.update(parsed)
            loaded.pop('config')
            parsed = loaded
            

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
            config_key = parsed['config']
            loaded = utils.predefine.load(key=config_key , yaml_filename='model')
            
            loaded.update(parsed)
            loaded.pop('config')
            parsed = loaded
            

        net_name = parsed['_key']
        model_name = parsed['model']
        
        model_args = inspect.getfullargspec( model.get_model_dict(lowercase=True)[model_name.lower()].__init__  ).args
        model_config = _get_filtered_dict(parsed,model_args)

        result_dict[net_name] = model.get(model_name=model_name , model_config=model_config).to(args['device'])
        
        if 'load' in parsed:
            load_path = parsed['load']
            sdict = torch.load(load_path)
            try:
                if "IS_TAEKLEARNING_CHECKPOINT" in sdict:
                    result_dict[net_name].load(sdict['MODEL_'+net_name])
                else:   
                    result_dict[net_name].load(sdict)
            except Exception as e:
                print(f"Load Failed for [{net_name}]\n")
                print(e)
            else:
                print(f"Successfully loaded [{net_name}] from {load_path}\n")

    args['model'] = result_dict
            
    

def prepare_trainchunk(args:dict):
    result_list = []
    dictionarylike_statements = args.get('trainchunk',[])

    for dictionarylike_statement in dictionarylike_statements:
        
        parsed = utils.dictionarylike.parse(dictionarylike_statement)

        if len(parsed)==1 and '_key' in parsed:
            parsed['config']=parsed['_key']

        if 'config' in parsed:
            config_key = parsed['config']
            loaded = utils.predefine.load(key=config_key,yaml_filename='trainchunk')
            
            loaded.update(parsed)
            loaded.pop('config')
            parsed = loaded
        
        if '_key' in parsed: parsed.pop('_key')
        
        result_list.append(parsed)

    args['trainchunk'] = result_list