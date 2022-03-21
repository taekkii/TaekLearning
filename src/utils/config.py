
import sys
import os

import task

def get_task_dict():
    return { task_name:eval( "task."+task_name.capitalize()+"Task" ) for task_name in task.get_all_tasks_names() }

#{taskname:class of task}

task_dict = get_task_dict()
