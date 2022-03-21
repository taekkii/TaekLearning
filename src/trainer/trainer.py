
from typing import Iterable
from typing import Union
from typing import Callable
from typing import Optional


from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim

from . import optimizer
from . import lrscheduler

import time


class Trainer:

    class TrainChunk:
        def __init__(self,net:Union[nn.Module,nn.parallel.DataParallel],
                          optimizer:Union[str,optim.Optimizer],
                          lr_scheduler:Union[str,optim.lr_scheduler._LRScheduler],
                          obj_func:Callable,
                          hyperparam:dict):
            self.net = net
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
            self.obj_func = obj_func
            self.hyperparam = hyperparam

        def prepare(self):
            
            if isinstance(self.optimizer,str):
                self.optimizer = optimizer.get_optimizer(self.net, self.optimizer, self.hyperparam)
            
            if isinstance(self.lr_scheduler,str):
                self.lr_scheduler = lrscheduler.get_lr_scheduler(self.net,self.lr_scheduler,self.hyperparam)
        
        def get_lr(self):
            return self.optimizer.param_groups[0]['lr']
        
        def forward_and_backward(self,*input,target:Optional[torch.Tensor],optimizer_step=True,lr_scheduler_step=True):
            
            self.net.train()

            output = self.net(*input)
            if target is not None:
                loss = self.obj_func(output,target)
            else:
                loss = self.obj_func(output)
            
            loss.backward()
            
            if 'grad_clip' in self.hyperparam:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(),self.hyperparam['grad_clip'])
            
            if optimizer_step:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            if lr_scheduler_step:
                self.lr_scheduler.step()

            return loss.item()
            

    def __init__(self,datasets,settings:dict,train_chunks:Iterable[TrainChunk]):
        
        #-----[WELCOME]-----#
        self.welcome()

        #-----[Propagate to self]-----#
        self.datasets     = datasets
        self.settings     = settings
        self.train_chunks = train_chunks

        #-----[Prepare train chunk]-----#
        for train_chunk in train_chunks:
            train_chunk.prepare()
        
        self.elapsed_second = 0
        self.time0 = time.time()

    def welcome(self):
        pass



    def _get_elapsed_second(self):
        time_delta = time.time() - self.time0
        self.time0 = time.time()
        self.elapsed_second += time_delta
        return self.elapsed_second