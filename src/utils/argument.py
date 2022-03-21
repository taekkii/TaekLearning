

import argparse

from . import config

def get_args():
    parser = argparse.ArgumentParser(description="Taekki's Neural Network Script")
    
    subparsers = parser.add_subparsers(help='Choose one of the tasks')

    for task_name,task_class in config.task_dict.items():
        sub_parser = subparsers.add_parser( name=task_name , help=f': Task' )
        
        if hasattr(task_class,"set_parser"):
            task_class.set_parser(parser=sub_parser)
    
    arg = parser.parse_args()
    
    return arg.__dict__
