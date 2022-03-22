
import sys
import os

import task
import dataset

def get_task_dict():
    return { task_name:eval( "task."+task_name.capitalize()+"Task" ) for task_name in task.get_all_tasks_names() }
#{taskname:class of task}

def get_dataset_dict():
    return { dataset_name:eval("dataset."+dataset_name+"Dataset") for dataset_name in dataset.get_all_dataset_names() }

def get_model_dict():
    pass

