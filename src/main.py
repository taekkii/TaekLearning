


import utils.argument
import utils.predefine
import utils.prepare
import utils.dictionarylike

def activate_predefine(target_list:list,save_filename:str):
    
    for target in target_list:

        parsed = utils.dictionarylike.parse(target)
        key = parsed['_key']
        parsed.pop('_key')

        utils.predefine.save(key,parsed,save_filename)
    

def main():
    arg = utils.argument.get_args()
    if arg['predefine_save']:
        activate_predefine(arg['dataset'],'dataset')
        activate_predefine(arg['model'],'model')
        exit()
    
    

    utils.prepare.prepare_dataset(arg)
    utils.prepare.prepare_model(arg)
    utils.prepare.prepare_trainchunk(arg)
    print(arg)

    task = arg['task_class'](arg)    
    task()

    
    
if __name__ == "__main__":
    main()