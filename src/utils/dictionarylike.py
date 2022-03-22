


def parse(input_string:str , split_token=' ', key_val_token='='):
    tokens = input_string.split(split_token)

    if key_val_token not in tokens[0]:
        tokens[0] = '_key'+key_val_token+tokens[0]   
    
    parse_dict = {}
    for token in tokens:
        k,v = token.split(key_val_token)
        parse_dict[k]=v

    return parse_dict
