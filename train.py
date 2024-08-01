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
    device = None
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
        observation, img, _, _ = env.reset()
        current_ep_reward = 0
        update_steps = 0
        while True:
            start = time.time()

            old_observation,old_img = observation, img

            action, action_logprob, state_val = ppo.select_action(observation, img)
            action = action.item()

            observation, img, reward, done = env.step(action)

            ppo.buffer.append(old_observation, old_img, action, reward, observation, img, action_logprob, done, state_val)
            
            update_steps+=1
            current_ep_reward += reward
            if not update_steps%100:
                # print(ppo.buffer.return_tensors())
                if ppo.buffer.update():
                    # import code; code.interact(local=locals())
                    loss, lr, norm = ppo.update()
                    
                    prog.log({
                                'RewardAvg':current_ep_reward/update_steps,
                                "loss":loss,
                                "lr":lr,
                                'norm':norm,
                                "buffer_siz"
                                "e":ppo.buffer._curr_size
                                })
                # import code; code.interact(local=locals())
            if not update_steps %2000:
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
    env.stop()

if __name__ == "__main__":
    parser = HfArgumentParser(TRAINS)
    
    (*parsed_args, unknown_args) = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    project_name = "AutoRouter"
    modelargs, trainingargs = parsed_args
    envargs = EnvArgs()
    train_iteration(modelargs, trainingargs, envargs, project_name=project_name)
