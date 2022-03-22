

import argparse

from . import config



import task


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
                                      [OR] 
                                    --model "config=net"
                                    
                                    ''')
    
    parser.add_argument('--trainchunk','-t',
                        nargs='+',
                        default=[],
                        help='''(Dictionary-like-form of) Train Chunk(set of NeuralNet, optimizer and possibly LRscheduler, with necessary hyperparameters). 
                                Ex) --trainchunk "net=classifier optimizer=SGD lrscheduler=CosineLR lr=30 momentum=0.9 weight_decay=1e-4 ..."''')

    parser.add_argument('--save','-s',
                        action='store_true',
                        help='Save all models after running the script is done.')  

    parser.add_argument("--disable-recent_checkpoint",
                         action='store_true',
                         help='The model saves checkpoint for every iteration, which happens for minimum cycle of 30 seconds. Using this flag will disable this utility.')

    parser.add_argument("--checkpoint-cycle",
                        type=int,
                        default=0,
                        help='The model saves checkpoint for every [CHECKPOINT-CYCLE]th iteration. Does nothing when [CHECKPOINT-CYCLE]=0(defualt)')
                    
    parser.add_argument('--predefine-save','-p',
                        action='store_true',
                        help='Saves all dictionary-like-form to provided PATH, and exits.')

    parser.add_argument("--epoch",'-e',
                       default=None,
                       help='The number of epochs. Error if used with --iter')

    parser.add_argument("--iter",'-i',
                        default=None,
                        help='The number of iterations. Error if used with --epoch')
    
    parser.add_argument('--flag','-f',
                        type=str,
                        help='Feel-free-to-use additioanl flag for your scripting.')
    
    

def get_args():
    parser = argparse.ArgumentParser(description="Taekki's Neural Network Script")
    
    subparsers = parser.add_subparsers(help='Choose one of the tasks')

    
    task_dict = task.get_task_dict()

    for task_name,task_class in task_dict.items():
        sub_parser = subparsers.add_parser( name=task_name , help=f': Task' )
        sub_parser.set_defaults(task_class=task_class)
        
        _default_args(sub_parser)
        
        try:
            task_class.set_parser(parser=sub_parser)
        except NotImplementedError:
            print(f"[WARNING] set_parser() is not implemented for task [{task_name}]")
    
    arg = parser.parse_args()
    
    argdict = arg.__dict__
    
    assert argdict['iter'] is None   or    argdict['epoch'] is None , "Both of # of iterations and # of epochs are given."
    
    
    return argdict