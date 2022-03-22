
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import torch.nn as nn
from typing import Optional
from notmine.katsura import CosineAnnealingWarmupRestarts

#-----[Supporting LR Scheduler]-----#
lr_Scheduler_dict = {
    'constantlr':dict( name="ConstantLR",
                       base_class=lr_scheduler.ConstantLR , 
                       flexible=[],
                       default=dict(total_iters=1,factor=1.0,last_epoch=-1) ),
    
    
    'steplr':dict( name="StepLR",
                   base_class=lr_scheduler.StepLR , 
                   flexible=['step_size','gamma'],
                   default=dict() ),
    
    
    'multisteplr':dict( name="MultiStepLR",
                        base_class=lr_scheduler.MultiStepLR, 
                        flexible=['milestones','gamma'],
                        default=dict() ),

    'simplelinearlr':dict( name="SimpleLinearLR",
                           exceptional=True ),

    'onecyclelr':dict( name="OneCycleLR",
                       exceptional=True),

    'cosinelr':dict( name="CosineLR",
                     exceptional=True),
    
    'cosineonecycle':dict( name='CosineOneCycle',
                           exceptional=True)

}
def get_lr_scheduler(optimizer:optim.Optimizer,
                     lr_scheduler_name:Optional[str],
                     lr_scheduler_hyperparam:dict):
    
    if lr_scheduler_name is None  or  lr_scheduler_name == '':
        lr_scheduler_name = 'constantlr'
    
    assert lr_scheduler_name.lower() in lr_Scheduler_dict   ,   f"Unsupporting LR scheduler:[{lr_scheduler_name}]"
    
    lr_scheduler_name = lr_scheduler_name.lower()

    if "exceptional" not in lr_Scheduler_dict[lr_scheduler_name]:
        hyperparams = {k:v for k,v in lr_scheduler_hyperparam.items() if k in lr_Scheduler_dict[lr_scheduler_name]['flexible'] }
        hyperparams.update(lr_Scheduler_dict[lr_scheduler_name]['default'])
        return lr_Scheduler_dict['base_class'](optimizer=optimizer,**hyperparams)
    else:
        if   lr_scheduler_name == 'simpllinearlr':
            return lr_scheduler.OneCycleLR(optimizer,
                                           max_lr=lr_scheduler_hyperparam['lr'],
                                           total_steps=lr_scheduler_hyperparam['iters'],
                                           pct_start=lr_scheduler_hyperparam['warmup_iters']/lr_scheduler_hyperparam['iters'],
                                           anneal_strategy='linear',
                                           div_factor=1e9,
                                           final_div_factor=1e9)
        
        elif lr_scheduler_name == 'onecyclelr':
            return lr_scheduler.OneCycleLR(optimizer,
                                           max_lr=lr_scheduler_hyperparam['lr'],
                                           total_steps=lr_scheduler_hyperparam['iters'])
        
        elif lr_scheduler_name == 'cosinelr':
            return CosineAnnealingWarmupRestarts(optimizer=optimizer,
                                                 first_cycle_steps=lr_scheduler_hyperparam['t_0'],
                                                 cycle_mult=lr_scheduler_hyperparam['t_mult'],
                                                 max_lr=lr_scheduler_hyperparam['lr'],
                                                 min_lr=lr_scheduler_hyperparam['eta_min'],
                                                 warmup_steps=lr_scheduler_hyperparam['warmup_iters'],
                                                 gamma=lr_scheduler_hyperparam['gamma'])
        elif lr_scheduler_name == 'cosineonecycle':
            return CosineAnnealingWarmupRestarts(optimizer=optimizer,
                                                 first_cycle_steps=lr_scheduler_hyperparam['iters'],
                                                 cycle_mult=1.0,
                                                 max_lr=lr_scheduler_hyperparam['lr'],
                                                 min_lr=lr_scheduler_hyperparam['eta_min'],
                                                 warmup_steps=lr_scheduler_hyperparam['warmup_iters'],
                                                 gamma=1.0)
    
