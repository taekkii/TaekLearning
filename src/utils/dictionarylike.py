


def parse(input_string:str , split_token=' ', key_val_token='=',cast_token='%',auto_cast = True):
    tokens = input_string.split(split_token)

    if key_val_token not in tokens[0]:
        tokens[0] = '_key'+key_val_token+tokens[0]   
    
    parse_dict = {}
    for token in tokens:
        k,v = token.split(key_val_token)
        if cast_token in v:
            vtmp,typ = v.split(cast_token)
            
            if typ in ['int','d']:
                v=int(vtmp)
            elif typ in ['float','f']:
                v=float(vtmp)
        elif auto_cast:
            if v == 'True' or v == 'False':
                v = v=='True'
            else:
                try:
                    v=int(v)
                except ValueError:
                    try:
                        v=float(v)
                    except ValueError:
                        pass

        
        parse_dict[k]=v

    return parse_dict
