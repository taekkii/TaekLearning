

import os
from pathlib import Path
import shutil

def prepare_dir(dir_path:str):
    """
    if a directory path(nested or not) not exists, creates it.
    [ARGS]
        dir_path: target path
    [RETRUNS]
        none
    """
    if os.path.exists(dir_path): return
    Path(dir_path).mkdir(parents=True,exist_ok=True)

def clear_folder(file_path:str):
    x = input(f"Are you really sure about clearing [{file_path}]?[y/n] ")
    if x.lower() != 'y': return

    shutil.rmtree(file_path)
    os.mkdir(file_path)
    print(f"{file_path} has been cleared")