
import os
import importlib
import sys



class Task:
    
    @classmethod
    def set_parser(cls,parser):
        raise NotImplementedError(f"Your task class [{cls.__name__}] has not implemented set_parser(parser) method")


