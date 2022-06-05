
from . import parser
from .. import Task




class NerfTask(Task):
    
    set_parser = parser.set_parser
    
    def __init__(self,args):
        
        super().__init__(args)
        
        print('=========== [NeRFTask] ==========')
        print("REALITY nogada dream comes true!")
        print("=================================\n\n")


        

    def __call(self):
        pass