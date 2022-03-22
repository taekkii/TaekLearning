
from . import parser
from .. import Task
import trainer

class ClassificationTask(Task):
    
    set_parser = parser.set_parser
    def __init__(self,args):
        super().__init__(args)
        

