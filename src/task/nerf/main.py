
from . import parser
from .. import Task

from .train import NeRFTrainer


class NerfTask(Task):
    
    set_parser = parser.set_parser
    
    def __init__(self,args):
        
        super().__init__(args)
        
        print('=========== [NeRFTask] ==========')



    def __call__(self):
        t = NeRFTrainer(self.datasets,self.models,self.trainchunks,self.settings)
        t.run()
        