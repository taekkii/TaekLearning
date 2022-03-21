


from typing import Union
import torch.nn as nn
import torch.optim as optim


#----[Supporting Optimizers]-----#

optimizer_dict = {
    'sgd':(optim.SGD, ["lr","momentum","weight_decay","nesterov"] ),
    'adam':(optim.Adam, ["lr","weight_decay","eps","betas"]),
    'adamw':(optim.AdamW, ["lr","weight_decay","eps","betas"] )
}



#-----[Get an optimizer from an optimizer name]-----#
def get_optimizer(net:Union[nn.Module,nn.parallel.DataParallel],
                  optimizer_name:str,
                  optimizer_hyperparameters:dict):
    assert optimizer_name.lower() in optimizer_dict , f"Unsupported optimizer {optimizer_name}"
    
    optimizer_cls , optimizer_hyperparam_keys = optimizer_dict[optimizer_name.lower()]
    hyperparams = {k:v for k,v in optimizer_hyperparameters.items() if k in optimizer_hyperparam_keys}
    return optimizer_cls(params=net.parameters() , **hyperparams)

