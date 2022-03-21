

import argparse

from . import config


def get_args():
    parser = argparse.ArgumentParser(description="Taekki's Neural Network Script")
    
    subparsers = parser.add_subparsers(help='Choose one of the tasks')

    parser.add_argument('--device','-d',
                        required=True,
                        help = '[Required] Device(s) for run')

    parser.add_argument('--dataset', '-ds',
                        type=list,
                        nargs='+',
                        help='Name of Dataset(s)')
    
    parser.add_argument('--model-parameter' , '-mp',
                        type=str,
                        help='''(Dictionary-like-form of) model parameters. Ex) -mp "layers=10, channels=3" ''')
    
    parser.add_argument('--flag','-f',
                        type=str,
                        help='Feel-free-to-use additioanl flag for your scripting.')
                        
    for task_name,task_class in config.task_dict.items():
        sub_parser = subparsers.add_parser( name=task_name , help=f': Task' )
        
        if hasattr(task_class,"set_parser"):
            task_class.set_parser(parser=sub_parser)
    
    arg = parser.parse_args()
    
    return arg.__dict__
