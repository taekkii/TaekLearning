
import json
import yaml

def get(key:str , config_file_path = "./config/default.json"):
    with open(config_file_path,'r') as fp:
        return json.load(fp)[key]

