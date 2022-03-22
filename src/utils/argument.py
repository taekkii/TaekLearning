

import argparse

from . import config
from typing import Iterable

import inspect 


import task
import dataset
import dataset.transform

import model
import utils.dictionarylike

from typing import Iterable

def _get_filtered_dict(original_dict:dict , include:Iterable[str] ):    
    return{k:v for k,v in original_dict.items() if k in include}


def _default_args(parser):

    parser.add_argument('--device','-d',
                        type=str.lower,
                        choices=config.get(key='device'),
                        required=True,
                        help = '[Required] Device(s) for run')

    parser.add_argument('--dataset', '-ds',
                        nargs='+',
                        default=[],
                        help='''(Dictionary-like-form of) Dataset(s)
                                Ex) --dataset "CIFAR10 transform=tf0" ''')
    
    parser.add_argument('--model' , '-m',
                        nargs='+',
                        default=[],
                        help='''(Dictionary-like-form of) model and parameters. 
                                Ex) --model "classifier (load=./saved/model.pth) model=ResNet layers=18, channels=3"
                                             OR 
                                            "classifier config=./predefined/resnet_settings.yaml" configkey=resnet18 ''')

    
    
    parser.add_argument('--flag','-f',
                        type=str,
                        help='Feel-free-to-use additioanl flag for your scripting.')

    

def _prepare_dataset(args:dict):

    result_dict = {}
    dictionarylike_statements = args.get('dataset',[])

    for dictionarylike_statement in dictionarylike_statements:
        parsed = utils.dictionarylike.parse(dictionarylike_statement)
        
        if 'config' in parsed:
            pass
        if 'transform' in parsed:
            parsed['transform'] = dataset.TRANSFORM_DICT[parsed['transform']]
        
        dataset_name = parsed['_key']
        
        
        dataset_args = inspect.getfullargspec( dataset.get_dataset_dict(lowercase=True)[dataset_name.lower()].__init__ ).args
        dataset_config = _get_filtered_dict(parsed,dataset_args)

        result_dict[dataset_name] = dataset.get(dataset_name=dataset_name , dataset_config=dataset_config)

    args['dataset'] = result_dict



def _prepare_model(args:dict):
    result_dict = {}
    dictionarylike_statements = args.get('model',[])
    
    for dictionarylike_statement in dictionarylike_statements:
        
        parsed = utils.dictionarylike.parse(dictionarylike_statement)

        if 'config' in parsed:
            pass

        model_name = parsed['_key']
        model_args = inspect.getfullargspec( model.get_model_dict(lowercase=True)[model_name.lower()].__init__  ).args
        model_config = _get_filtered_dict(parsed,model_args)

        result_dict[model_name] = model.get(model_name=model_name , model_config=model_config)

    args['model'] = result_dict
            
def get_args():
    parser = argparse.ArgumentParser(description="Taekki's Neural Network Script")
    
    subparsers = parser.add_subparsers(help='Choose one of the tasks')

    
    task_dict = task.get_task_dict()

    for task_name,task_class in task_dict.items():
        sub_parser = subparsers.add_parser( name=task_name , help=f': Task' )
        sub_parser.set_defaults(task_classname=task_class)
        
        _default_args(sub_parser)
        
        try:
            task_class.set_parser(parser=sub_parser)
        except NotImplementedError:
            print(f"[WARNING] set_parser() is not implemented for task [{task_name}]")
    
    arg = parser.parse_args()
    
    argdict = arg.__dict__
    
    #-----[Post-Processes]-----#

    _prepare_dataset(argdict)
    _prepare_model(argdict)

    return argdict