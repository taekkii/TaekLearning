
from typing import Iterable
from typing import Union
from typing import Callable
from typing import Optional

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim

from . import optimizer
from . import lrscheduler
from . import visualizer

import time


from tqdm import tqdm,trange


NUM_WORKERS = 10
GENERATOR_TOLERANCE = int(1e8)
EPOCH2ITER_KEYS_TRAINCHUNK = ['t_0','t_mult','warmup','milestones','step_size']
EPOCH2ITER_KEYS_SETTINGS = ['checpoint-cycle']

MIN_VISUALIZE_INTERVAL = 2.0
MIN_RECENT_CHECKPOINT_INTERVAL = 60.0

TQDM_MININTERVAL = 0.01
CHECKPOINT_KEYWORDS = ['settings','elapsed_second','history','it']
CHECKPOINT_DIR_PATH = './checkpoint'

TEST_DATASET_KEYWORDS = ['validation','val','test']


class Trainer:

    class TrainChunk:
        def __init__(self,trainchunk_dict:dict,models:Optional[Iterable[nn.Module]] = None):

            #-----[Alias(es)]-----#
            t = trainchunk_dict

            #-----[GUARD]-----#
            for key in ['net','optimizer'] :
                assert key in trainchunk_dict , f'Missing required key {key}'

            #-----[Propagate to self]-----#
            self.net_name = t['net']
            self.optimizer_name = t['optimizer']
            self.lr_scheduler_name = t.get('lr_scheduler',None)


            self.hyperparam = {k:v for k,v in trainchunk_dict.items() if k not in ['net','optimizer','lr_scheduler']}


            if models is not None:
                self.config_network(models)
            
        def config_network(self,models:dict):
            self.net = models[self.net_name]

        def config_optimizer(self):
            
            self.optimizer = optimizer.get_optimizer(self.net, self.optimizer_name, self.hyperparam)
            if self.lr_scheduler_name is not None:
                self.lr_scheduler = lrscheduler.get_lr_scheduler(self.optimizer,self.lr_scheduler_name,self.hyperparam)
        
        

        def get_lr(self):
            return self.optimizer.param_groups[0]['lr']
        

        def forward_and_backward(self,*input,target:Optional[torch.Tensor],obj_func:Callable,optimizer_step=True,lr_scheduler_step=True):
            
            assert hasattr(self,'net') , "Configure your model by calling config_network() first."
            assert hasattr(self,'optimizer') , 'config your optimizer(and possibly LR scheduler) by calling config_optimizer first.'
            self.net.train()

            output = self.net(*input)
            if target is not None:
                loss = obj_func(output,target)
            else:
                loss = obj_func(output)
            
            loss.backward()
            
            if 'grad_clip' in self.hyperparam:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(),self.hyperparam['grad_clip'])
            
            if optimizer_step:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            if lr_scheduler_step   and   self.lr_scheduler_name is not None:
                self.lr_scheduler.step()

            return output, loss.item()
      
        def state_dict(self):
            sdict = {'net':self.net_name , 'optimizer':self.optimizer.state_dict() , 'hyperparam':self.hyperparam }
            if hasattr(self,'lr_scheduler'):
                sdict['lr_scheduler'] = self.lr_scheduler.state_dict()
            return sdict

        def load_state_dict(self,sdict,models:Optional[dict]=None):            
            #this will not load the network parameter

            if models is not None:
                self.config_network(models)

            assert hasattr(self,'net') , "Configure your model by calling config_network() first."
            assert hasattr(self,'optimizer') , 'config your optimizer(and possibly LR scheduler) by calling config_optimizer first.'

            self.optimizer.load_state_dict(sdict['optimizer'])
            
            if hasattr(self,'lr_scheduler'):
                self.lr_scheduler.load_state_dict( sdict['lr_scheduler'] )
            
            self.hyperparam = sdict['hyperparam']

    class Addon:
        def __init__(self,func:Callable[[],None],cycle:int = 1,min_interval:Optional[float] = None , *args) :
            
            assert cycle > 0

            self.func = func
            self.cycle = cycle
            self.min_interval = min_interval
            self.args = args
            self.t0 = time.time()
            
        
        def __call__(self,iter):

            if self.min_interval is not None  and  time.time() - self.t0 <= self.min_interval:
                return
            self.t0 = time.time()

            if iter % self.cycle == 0:
                self.func(*self.args)
        
    
        
                                
    def __init__(self,datasets:dict , models:dict , trainchunks:Iterable[Union[dict,TrainChunk]] , settings: dict):
        
        #-----[Propagate to self]-----#
        self.datasets     = datasets
        self.models       = models
        self.settings     = settings


        #-----[Prepare Dataloader]-----#
        self.get_dataloader()
        
        #-----[Convert Epoch => Iterations]-----#
        self.convert_epoch_to_iterations(trainchunks)

        #-----[Configure Train Chunk]-----#
        self.trainchunks  = self._config_trainchunk(trainchunks)
        

        #-----[WELCOME]-----#
        self.welcome()
        
        #-----[Prepare History]-----#
        self.history = {}

        #-----[Clock Reset]-----#
        self.elapsed_second = 0
        self.time0 = time.time()
        
        #-----[Iteration Reset]-----#
        self.it = 1

        #-----[Addons]-----#
        self.addons = []
        
        if self.settings['checkpoint_cycle'] > 0:
            self.add_on(self._chkpoint_cycle,cycle=self.settings['checkpoint_cycle'])
        
        if not self.settings['disable_recent_checkpoint']:
            self.add_on(self._chkpoint_recent,min_interval=MIN_RECENT_CHECKPOINT_INTERVAL)

        #-----[Resume if Needed]-----#
        if self.settings['resume'] is not None:
            resume_file = self.settings['resume']
            self.resume(resume_file)

        #-----[Prepare Visualizer]-----#
        self.visualizer = visualizer.Visualizer(env=self.settings['experiment_name'])
        self.add_on(self.visualizer.run,min_interval=MIN_VISUALIZE_INTERVAL)
        self.start()

        
        

    def add_on(self,method,cycle:int = 1,min_interval:Optional[float] = None,*arg):
        self.addons.append(Trainer.Addon(method,cycle,min_interval,*arg))

    def _chkpoint_cycle(self):
        self.save(checkpoint_file=f"{self.settings['experiment_name']}_{self.it}")

    def _chkpoint_recent(self):
        self.save(checkpoint_file=f"{self.settings['experiment_name']}_recent")

    
    def convert_epoch_to_iterations(self,trainchunks):
        
        #-----[GUARD]-----#
        if 'epoch' not in self.settings: return

        
        iters_per_epochs = sum( len(dataloader) for loader_name,dataloader in self.dataloaders.items() if loader_name not in TEST_DATASET_KEYWORDS )        
        
        #-----[Convert to Iter]-----#
        self.settings['iters_per_epochs'] = iters_per_epochs
        self.settings['iter'] = self.settings['epoch'] *  iters_per_epochs
        
        #-----[Update Settings]-----#
        for key in EPOCH2ITER_KEYS_SETTINGS:
            if key in self.settings:
                self.settings[key] *= iters_per_epochs

        
        #-----[Update LR Scheduler Hyperparameters]-----#
        for trainchunk in trainchunks:
            target_dict = trainchunk if isinstance(trainchunk,dict) else trainchunk.hyperparam

            for key in EPOCH2ITER_KEYS_TRAINCHUNK:
                if key in target_dict:
                    target_dict[key] *= iters_per_epochs
        


    def get_dataloader(self,shuffle=True):

        # Note : Default [Trainer] class sets SAME batch size for all dataloaders
        #        For different batch size for different datasets, ust override this function when you write subclass script.

        assert hasattr(self,'datasets')  ,  "No datasets found"
        assert 'batch_size' in self.settings  ,  "'settings' misses key [batch_size]"
        
        self.dataloaders = {k:DataLoader(dataset , self.settings['batch_size'] , shuffle=shuffle , num_workers=NUM_WORKERS) for k,dataset in self.datasets.items() }
            
    #generator
    def get_data(self,exception_keys:Iterable[str] = []):

        # Note : Default [Trainer] class just iterates over all datasets possessed.
        #        For different behavior, just override this generator when you write a subclass script.
        if not hasattr(self,'dataloaders'):
            self.get_dataloader()
        
        for _ in range(GENERATOR_TOLERANCE): #this is basically "infinite loop", but safer
            for name,dataloader in self.dataloaders.items():
                if name in exception_keys:continue
                
                for data in dataloader:
                    yield data 
    
    def start(self):
        pass
    
    def run(self):
        

        for data in tqdm( self.get_data(exception_keys=TEST_DATASET_KEYWORDS) , 
                          desc="Training Process", mininterval=TQDM_MININTERVAL , 
                          total=self.settings['iter'],initial = self.it-1):
            self.step(data)

            for addon in self.addons:
                addon(self.it)
            
            
            #-----[Iterate one step]-----#
            self.record("iter",self.it)
            self.it+=1

            if self.it > self.settings['iter']:
                break
    

    def record(self , key:str , value=None):
        if key not in self.history:
            self.history[key] = []
        if value is not None:
            self.history[key].append( value )
        
        return self.history[key]

    def resume( self , resume_file , dir_path=CHECKPOINT_DIR_PATH):
        
        path = dir_path + '/' + resume_file + '.pth'

        sdict = torch.load(path)

        assert 'TAEKLEARNING_CHECKPOINT' in sdict  ,  f"[{path}] : Not a TaekLearning Checkpoint File"


        for model_name,model in self.models.items():
            model.load_state_dict( sdict["MODEL_" + model_name] )

        for i,trainchunk in enumerate(self.trainchunks):
            trainchunk.load_state_dict( sdict['TRAINCHUNK_' + str(i)] ) 

        for key in CHECKPOINT_KEYWORDS: 
            setattr(self,key,sdict['TRAINERKEY_'+key])

        print(f"[Checkpoint Loaded] from {path} : [{self.it}]th iteration")
            
    def save(self, checkpoint_file , dir_path = CHECKPOINT_DIR_PATH):
        
        sdict = {'TAEKLEARNING_CHECKPOINT':True}

        path = dir_path + '/' + checkpoint_file + '.pth'
        
        for model_name,model in self.models.items():
            sdict["MODEL_" + model_name] = model.state_dict()

        for i,trainchunk in enumerate(self.trainchunks):
            sdict['TRAINCHUNK_' + str(i)] = trainchunk.state_dict()

        for key in CHECKPOINT_KEYWORDS: 
            sdict['TRAINERKEY_'+key] = getattr(self,key)

        torch.save(sdict,path)

        print(f"[Checkpoint Saved] to {path} : [{self.it}]th iteration")

    def step(self,*data):
        raise NotImplementedError("step() not implemented")
    

    def welcome(self):
        print("Welcome to Taekki's trainer")
        print("Dataset:",self.datasets)
        print("Models:",self.models)
        print("Train Chunks:",self.trainchunks)
        print("Settings:",self.settings)


    def _get_elapsed_second(self):
        time_delta = time.time() - self.time0
        self.time0 = time.time()
        self.elapsed_second += time_delta
        return self.elapsed_second
    
    def _config_trainchunk(self,trainchunks):

        ret_list = []
        
        for trainchunk in trainchunks:
            if isinstance(trainchunk,dict):
                t = Trainer.TrainChunk(trainchunk)
            else:
                t = trainchunk
            t.hyperparam['iter'] = self.settings['iter']

            t.config_network(self.models)
            t.config_optimizer()
            ret_list.append(t)
            
        return ret_list

    