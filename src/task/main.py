
class Task:
    
    def __init__(self,args:dict):
        self.dataset = args['dataset']
        self.model   = args['model']
        self.device  = args['device']
        self.train_chunk = args['trainchunk']
        
        self.epoch = args['epoch']
        self.iter  = args['iter']
        
        
#        print(self.dataset)
#        print(self.model)
#        print(self.device)
#        print(self.train_chunk)
        

    @classmethod
    def set_parser(cls,parser):
        raise NotImplementedError(f"Your task class [{cls.__name__}] has not implemented set_parser(parser) method")


    def __call__(self):        
        raise NotImplementedError(f"Your task class [{self.__class__.__name__}] has not implemented __call__(self) method")