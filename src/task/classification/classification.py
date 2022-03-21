
from . import parser
from ..task import Task

import dataset


class ClassificationTask(Task):
    
    set_parser = parser.set_parser
    def __init__(self,args):
        pass