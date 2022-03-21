

import argparse

@classmethod
def set_parser(cls,parser:argparse.ArgumentParser):

    ########## Basic Arguments ########## 
    parser.add_argument('--train',
                        action = 'store_true',
                        help = "Use the script to train.")

    parser.add_argument('--test',
                        action = 'store_true',
                        help = "Use the script to test. You need to provide --load argument additionally.")
    
    parser.add_argument('--model-summary','-ms',
                        action='store_true',
                        help="Show summary of model before any processes")

    ########## Path Arguments ###########
    parser.add_argument('--save','-s',
                        default = 'trained_model',
                        help = "FILENAME for saving your trained model, 'trained_model' as default\n"+
                               "The model will be saved at './saved/[SAVE_PATH]\n"+
                               "Additionally, your Visdom server will display the training process at environment titled '[SAVE]'." )
    
    parser.add_argument('--load','-l',
                        default = None,
                        help = "ENTIRE PATH for loading your trained model.\n[None] as default, in which case the model starts training from its initial state\n"+
                               "If you are using the script only for test and not loading any model parameters, you are very likely to doing something meaningless.\n")
    
    parser.add_argument('--resume','-r',
                        default = None,
                        help = "Resume from your checkpoint. Filename only is necessary(Don't use completed path). --load will be ignored.")

    
    parser.add_argument('--save-on-better',
                        action = 'store_true',
                        help = 'Saves model on better (validation)accuracy.')
                        
    ########## 'Training-related' Arguments ##########
    parser.add_argument("--learning-rate",'--lr',
                        type = float,
                        default = 0.1,
                        help = '(Initial) Learning Rate')

    parser.add_argument("--batch-size",
                        type = int,
                        default = 128,
                        help = 'Batch Size')
    
    parser.add_argument('--epoch','--ep',
                        type = int,
                        default = 20,
                        help = 'Epoch')

    parser.add_argument("--checkpoint-cycle",
                        type = int,
                        default = 0,
                        help = "The model will save training checkpoint for every [checkpoint-cycle]th epoch.\n"+
                               "Default = 0(no cycle checkpoint)")
    
    parser.add_argument('--optimizer','--optim',
                        choices = ('SGD','Adam','AdamW'),
                        default = 'SGD',
                        help = 'Optimizer for your trainer. Default = (torch.optim.)SGD.\n'+
                               'Currently available : SGD, Adam, Adamw')
                               
    parser.add_argument('--lr-scheduler',
                        default='none',
                        nargs="+",
                        help = "Choose from(caps-free, muiltiselect): \n" +  
                               "'None', no parameters\n" +
                               "'StepLR', give --step-size and --gamma\n"
                               "'MultistepLR', give --milestones as list and --gamma\n"+
                               "'SimpleLinearLR'. give no parameters.\n"
                               "'OneCycleLR', no parameters\n"+
                               "'CosineAnnealingLR', give --t-max(--t-0) and --eta-min(default:0)\n" +
                               "'CosineAnnealingWarmRestarts', give --t-0, --t-mult(default:1) and --eta-min(default:0)\n"+
                               "'CosineAnnealingWarmUPRestarts', give --t-0, --t-mult(default:1), --eta-min(default:0)\n"+
                               "  and --warmup-epochs(default:0)\n"+
                               "'CosineOneCycle'. give --warmup-epochs(default 0) and --eta-min(default 0) if want.\n"+
                               " Basically equivalent to: \n"+
                               " CosineAnnealingWarmupRestarts with [T_0] = [TOTAL_EPOCHS]\n"+
                               "\n")
    ######### Optimizer hyperparameters ##########

    parser.add_argument('--weight-decay',
                        type=float,
                        default = 0.,
                        help = "Weight Decay")
    
    parser.add_argument('--momentum',
                        type = float,
                        default = 0.,
                        help = "Momentum. Has no effect when applying Adam or AdamW as an optimizer")
    parser.add_argument('--grad-clip',
                        type=float,
                        default = -1.0,
                        help="Grad Clip. Has no effect if GRAD-CLIP<0(default:-1)")
    parser.add_argument('--nesterov', 
                        action='store_true',
                        help='Applies nesterov momentum. Only works on SGD')

    ########## LR-Scheduler hyperparameters ###########
    parser.add_argument('--step-size',
                        type=int,
                        help="Only for StepLR scheduler")
    
    parser.add_argument('--gamma',
                        type=float,
                        default=0.1,
                        help="Only for StepLR AND MultiStepLR scheduler. Default 0.1")
    
    parser.add_argument('--milestones',
                        nargs='+',
                        help = 'Only for MultiStepLR scheduler\n'+
                               'Usage: (ex)--milestones 10 15 20' )

    parser.add_argument('--t-0','--t0','--t-max','--tmax',
                        type=int,
                        help="Only for families of CosineAnnealingLR scheduler\n"+
                             "NOTE: It will give optimizer-step(the lr WILL change for each iteration),\n"+
                             "      but one cycle is [T_0] 'epoch', not [T_0] 'iteration'\n"+
                             "      Updated: [2022/02/27].")

    parser.add_argument('--t-mult','--tmult',
                        type=float,
                        default=1.,
                        help="Only for CosineAnnealingWarm(up)RestartsLR scheduler. Default 1")
                        
    parser.add_argument('--eta-min',
                        type=float,
                        default=0.0,
                        help="Only for families of CosineAnnealingLR schedler. Default 0")
    
#     parser.add_argument('--eta-max',
#                         type=float,
#                         default=0.01,
#                         help="Only for CosineAnnealing'WARMUP'LR schedler. Default 0.01")
    
    parser.add_argument('--warmup-epochs',
                        type=int,
                        default=0,
                        help="Only for CosineAnnealing'WARMUP'LR schedler. Default 0")
    
    
    
    
    ########### Etc. ##########
    parser.add_argument('--device',
                        default = None,
                        help = 'Forces your device to be [DEVICE]. Recommend not to use.')
                        
    parser.add_argument('--disable-dataparallel',
                        action='store_true',
                        help = 'Use this flag to avoid using DataParallel')

    parser.add_argument('--flag','-f',
                        default='',
                        help = 'Feel-free-to-use flag for scripting.')

    return parser