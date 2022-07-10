


import utils.argument
import utils.predefine
import utils.prepare
import utils.dictionarylike
import utils.config
import utils.system

import torch

import dataset
import model
import task
import os

def record_predefine(target_list:list,save_filename:str):
    
    for target in target_list:
        
        parsed = utils.dictionarylike.parse(target)
        
        if 'config' in parsed  or  len(parsed)<=1: continue
        
        key = parsed['_key']
        parsed.pop('_key')

        utils.predefine.save(key,parsed,save_filename)


def summary_predefines():
    for filename in ['dataset','model','trainchunk']:
        dic = utils.predefine.load(yaml_filename=filename)
        
        print(f"[{filename}]")
        for k,vdic in dic.items():
            print('-',k)
            for k2,v in vdic.items(): print(' >',k2,':',v)
            
    
def summary_available_settings():

    print("\n[Datasets]")
    for dataset_name in dataset.get_all_dataset_names(): print(dataset_name)

    print('\n[Models]')
    for model_name in model.get_all_model_names():print(model_name)

    print('\n[Tasks]')
    for task_name in task.get_all_tasks_names() :print(task_name)
    

def main():
    arg = utils.argument.get_args()
    
    if arg['predefine_save']:
        record_predefine(arg['dataset'],'dataset')
        record_predefine(arg['model'],'model')
        record_predefine(arg['trainchunk'],'trainchunk')
        exit()
    elif arg['clear_checkpoint']:
        utils.system.clear_folder('./checkpoint')
        exit()
    elif arg['summary_predefines']:
        summary_predefines()
        exit()
    elif arg['summary_available_settings']:
        summary_available_settings()  
        exit()      
    
    #-----[RANDOM_SEED]-----#
    random_seed = utils.config.get("random_seed")
    torch.manual_seed( random_seed )
    if 'cuda' in arg['device']: 
        torch.cuda.manual_seed( random_seed )


    utils.prepare.prepare_dataset(arg)
    utils.prepare.prepare_model(arg)
    utils.prepare.prepare_trainchunk(arg)
    
    
    t = arg['task_class'](arg)    
    t()

    if arg['save']:
        for model_name , net in arg['model'].items():
            path = utils.config.get('save_path')
            utils.system.prepare_dir(path)
            file_path = os.path.join(path,f"{arg['experiment_name']}_{model_name}.pth")
            
            torch.save(net,file_path)
            print(f'[SAVED MODEL] to {file_path}')
    
if __name__ == "__main__":
    main()