
import os


EXCPETIONAL_DIRECTORIES = ['__pycache__']


class Task:
    
    @classmethod
    def set_parser(cls,parser):
        raise NotImplementedError(f"Your task class [{cls.__name__}] has not implemented set_parser(parser) method")



def get_all_tasks_names()->list:
    
    file_path = os.path.realpath(__file__)
    dir_path = os.path.dirname(file_path)
    return [entry.name for entry in os.scandir(dir_path) if entry.is_dir()  and entry.name not in EXCPETIONAL_DIRECTORIES ]