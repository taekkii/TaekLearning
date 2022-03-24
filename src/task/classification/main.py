
from typing import Callable, Optional , Union, Iterable
from trainer import Trainer

from . import parser
from .. import Task
import trainer
import torch
import torch.nn as nn
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader

@torch.no_grad()
def get_classification_accuracy(dataloader:DataLoader , net:nn.Module , loss_fn:Optional[Callable] = None):
    
    correct = 0
    all = 0
    loss=0
    
    net.eval()
    device = next(net.parameters()).device
    
    for i, (x,t) in tqdm(enumerate(dataloader),desc="[Inference]",total=len(dataloader)):
        x = x.to(device)
        t = t.to(device)
        y = net(x)
        if loss_fn is not None:
            loss += loss_fn(y,t).item()
        
        test_vec = y.argmax(dim=-1)==t
        all += test_vec.shape[0]
        correct += test_vec.long().sum().item()
    
    
    if loss_fn:
        return correct/all , loss*len(dataloader) / all
    else:
        return correct/all

        
class ClassificationTask(Task):
    
    set_parser = parser.set_parser


    class ClassificationTrainer(trainer.Trainer):
        
        #override
        def start(self):
            self.visualizer.attach_graph('Iteration Loss',self.record('train_loss_iter'),'Train',x0=1)
          
            self.visualizer.attach_graph('Loss',self.record('train_loss'),'Train',x0=1)
            self.visualizer.attach_graph('Loss',self.record('val_loss'),'Validation',x0=1)
          
            self.visualizer.attach_graph('Accuracy',self.record('train_acc'),'Train',x0=1)
            self.visualizer.attach_graph('Accuracy',self.record('val_acc'),'Validation',x0=1)
            
            self.visualizer.attach_graph('Learning_Rate',self.record('lr'),'lr')
            self.correct , self.all = 0,0
            self.train_loss = 0
            
            self.add_on(self.validate,cycle=self.settings['iters_per_epochs'])

        #override
        def step(self,data):
            assert len(self.trainchunks)==1
            train_chunk = self.trainchunks[0]

            x,t = data
            x = x.to(self.settings['device'])
            t = t.to(self.settings['device'])

            y,train_loss = train_chunk.forward_and_backward(x,target=t,obj_func=torch.nn.CrossEntropyLoss(label_smoothing=0.1))
            
            self.correct += (y.argmax(-1) == t).long().sum().item()
            self.all += t.shape[0]
            self.train_loss += train_loss * t.shape[0]

            self.record('train_loss_iter',train_loss)
            self.record('lr',train_chunk.get_lr())
        
        def validate(self):

            loader = self.dataloaders['validation']
            val_acc,val_loss = get_classification_accuracy(loader,self.models['net'],torch.nn.CrossEntropyLoss(label_smoothing=0.1) )
            
            self.record('train_loss' , self.train_loss / self.all)
            self.record('train_acc'  , self.correct*100 / self.all)
            self.record('val_loss'   , val_loss)
            self.record('val_acc'    , val_acc*100)

            self.correct , self.all = 0,0
            self.train_loss = 0
            


    def __init__(self,args):
        super().__init__(args)
        
        self.settings    = copy.deepcopy(args)

        self.datasets    = self.settings.pop('dataset')
        self.models      = self.settings.pop('model')
        self.trainchunks = self.settings.pop('trainchunk')
        self.is_train = self.settings.pop("train")
        self.is_test  = self.settings.pop("test")


    def __call__(self):
        if self.is_train:
            t = ClassificationTask.ClassificationTrainer(self.datasets,self.models,self.trainchunks,self.settings)
            
            t.run()
        if self.is_test:
            
            loader = DataLoader(self.datasets['test'],batch_size=self.settings['batch_size'],shuffle=False,num_workers=10)
            acc = get_classification_accuracy(loader,self.models['net'])
            print(f"Accuracy:{acc*100:.2f}%")
    