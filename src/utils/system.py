

import os
import shutil


def clear_folder(file_path:str):
    x = input(f"Are you really sure about clearing [{file_path}]?[y/n] ")
    if x.lower() != 'y': return

    shutil.rmtree(file_path)
    os.mkdir(file_path)
    print(f"{file_path} has been cleared")