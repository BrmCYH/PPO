import wandb
import sys, os, time, math
from pathlib import Path
this_dir = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(str(Path(this_dir).parent))

from transformers import HfArgumentParser
import torch
from typing import Dict

from usages import POut, pad_dict, PackAndLog, load_model_from_disk, model_save_to_disk
from trainer import PPOTrain
from Agent import PPOAgent
from hyparam import Modelargs, Trainingargs, EnvArgs
from envs import ob_envs

TRAINS = [Modelargs, Trainingargs]


def train_iteration(modelargs: Modelargs, trainingargs:Trainingargs, envargs:EnvArgs, **kwargs):
    
    if modelargs.load_from_disk:
        config, state_dict = load_model_from_disk(modelargs.load_from_disk)
        modelargs = Modelargs(**config)
        trainingargs = Trainingargs(**config)
        POut(config)
    else:
        config = PackAndLog(modelargs, trainingargs)
        state_dict = None
    
    cache_dir = os.path.join(os.getcwd(),"cache_dir") if trainingargs.cache_dir is None else trainingargs.cache_dir
    ppo = PPOTrain(modelargs=modelargs, trainingargs=trainingargs, state_dict= state_dict)
    prog = wandb.init(
        project = kwargs["project_name"],
        name = f'{config["prog"]}',
        )
    # 
    env = ob_envs(envargs= envargs)
    avgrewardslist = []
    for t in range(1, trainingargs.episodes):
        observation, _,_ = env.reset()
        current_ep_reward = 0
        update_steps = 0
        while True:
            start = time.time()
            old_observation = observation
            action, action_logprob, state_val = ppo.select_action(observation)
            action = action.item()
            if envargs.apply_api_compatibility:
                observation, reward, done, truncated, _ = env.step(action)
            else:
                observation, reward, done, _ = env.step(action)
            ppo.buffer.append(old_observation, action, reward, observation, action_logprob, done, state_val)
            
            update_steps+=1
            current_ep_reward += reward
            if ppo.buffer.update():
                if done or (envargs.apply_api_compatibility and truncated): # 强制更新
                    loss, lr, norm = ppo.update()
                    
                    prog.log({
                                'RewardAvg':current_ep_reward/update_steps,
                                "loss":loss,
                                "lr":lr,
                                'norm':norm,
                                "buffer_size":ppo.buffer._curr_size
                                })
                    break
        reward_value = current_ep_reward/update_steps
        print(f"| RewardAvg:{reward_value:10.6} | Time exausted average: {time.time()-start:5.3}s |")
        
        if len(avgrewardslist) == 0:
            avgrewardslist.append(reward_value)
            static_dict = ppo.pioneer.state_dict()
        else:
            if avgrewardslist[0]> reward_value:
                avgrewardslist.append(reward_value)
            else:
                avgrewardslist.insert(0, reward_value)
                static_dict = ppo.pioneer.state_dict()
    model_save_to_disk(path =cache_dir, config=config, static_dict= static_dict)
                
if __name__ == "__main__":
    parser = HfArgumentParser(TRAINS)
    
    (*parsed_args, unknown_args) = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    project_name = "AutoRouter"
    modelargs, trainingargs = parsed_args
    envargs = EnvArgs()
    train_iteration(modelargs, trainingargs, envargs, project_name=project_name)
