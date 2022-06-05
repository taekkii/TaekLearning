
import copy

class Task:
    
    def __init__(self,args:dict):
        
        self.settings    = copy.deepcopy(args)

        self.datasets    = self.settings.pop('dataset')
        self.models      = self.settings.pop('model')
        self.trainchunks = self.settings.pop('trainchunk')

        

    @classmethod
    def set_parser(cls,parser):
        raise NotImplementedError(f"Your task class [{cls.__name__}] has not implemented set_parser(parser) method")


    def __call__(self):        
        raise NotImplementedError(f"Your task class [{self.__class__.__name__}] has not implemented __call__(self) method")