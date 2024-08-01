from datetime import datetime
from time import time
import os
import torch
from typing import Dict, Any
def pad_dict(parameterdict, **kwargs):
    
    for name, v in kwargs.items():
        if isinstance(v, dict):
            for i,j in v.items():
                assert not i in parameterdict.keys(), f'Parameter {i} has been created before'
                parameterdict[i]=[j]
        else:
            try:
                param =  v.__dict__
            except:
                raise f"The formate is correct. Parameters: {name}. "
            for i,j in param.items():
                assert not i in parameterdict.keys(), f'Parameter {i} has been created before'
                parameterdict[i]=[j]
    return parameterdict

def POut(parameters):
    start = "parameters"
    print(f"+{start:-^30}+")
        
    for k,v in parameters.items():
        try:
            item = v[0]
            if isinstance(v, list):
                
                print(f"|{k:30}:{v[0]:^30}|")
            else:
                print(f"|{k:30}:{v:^30}|")
        except TypeError:
            no = "None"
            print(f'|{k:30}:{no:^30}|')
    end = '-'
    print(f"+{end:-^30}+")

def PackAndLog(modelargs, trainingargs):
    parameters = {}
    parameters = pad_dict(parameters, modelargs = modelargs, trainingargs = trainingargs)
    parameters['maxsize' ]=[64]
    current_date = datetime.now().date()
    month = current_date.month
    day = current_date.day
    parameters['prog'] = f"proj_router_{month}_{day}_{str(int(time()))[-7:]}"

    POut(parameters)
    return parameters

def model_save_to_disk(path:str, config:Dict, static_dict: Any):
    local_path = os.path.join(path,f"{config['prog']}")
    try:
        os.mkdir(local_path)
    except:
        raise "Create floder Error!"
    with open(os.path.join(local_path,f"config.json"),'w') as f:
        import json
        json.dump(config, f)
    try:
        torch.save(static_dict, os.path.join(local_path,"ptmodel.pt"))
    except:
        raise "Model storage Error!"
def load_model_from_disk(path:str):
    assert os.path.isdir(path) == True,"Error, Path:{path} is not a floder"
    with open(os.path.join(path,"config.json"),'r')as f:
        import json
        configs = json.load(f)
    state_dict = torch.load(os.path.join(path,'ptmodel.pt'))
    return configs, state_dict
