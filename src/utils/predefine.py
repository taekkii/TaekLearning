

import yaml

from typing import Optional

def save(key,obj,yaml_filename='default',yaml_dir_path='./predefined'):

    file_path = yaml_dir_path+'/'+yaml_filename+'.yaml'
    try:
        with open(file_path,'r') as fp:
            di = yaml.load(fp,Loader=yaml.FullLoader)
    except FileNotFoundError:
        di = {}

    di[key] = obj


    with open(file_path,'w') as fp:
        yaml.dump(di,fp)

    print(f"Saved a predefine setting to {file_path}")


def load(key:Optional[str]=None , yaml_filename='default' , yaml_dir_path='./predefined'):

    file_path = yaml_dir_path+'/'+yaml_filename+'.yaml'

    try:
        with open(file_path,'r') as fp:
            di = yaml.load(fp,Loader=yaml.FullLoader)
    except FileNotFoundError:
        di = {}
        
    if key is None: return di
    else:           return di[key]
    