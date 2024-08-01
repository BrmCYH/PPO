import gymnasium as gym
import keyboard
import mss
import cv2
import time
import numpy as np
import pandas as pd
from gymnasium import Env
from hyparam import EnvArgs
from typing import Union
from abc import ABC
import torch
from utils import KeyboardScreenshot
import torchvision
import win32gui

ENVIRONMENT_NAMES= ["Genshin-ys"]
REGISTER_ENVS={}

class AbcEnv(ABC):
    def step(self):
        '''
            Return the environment callback based on action
        '''
    def reset(self):
        """
            Reset the environments
        """
    def stop(self):
        pass

class GenshinGames(AbcEnv):
    def __init__(self, envargs: EnvArgs):
        '''
        init EnvArgs Games
            help: Keep attention staying in Games
                init keyboard thread, snapshot thread 
                Building Connection between Games and database, eg.Redis, Mysql
            Args:
                EnvNames
                gym_id: if EnvNames havn't been supported, will execute game-env in gym space
                max_episode_steps: Epoch in each task.
                autoreset: Return to assigned place in each episode
                
        '''
        self.keyboardobj = KeyboardScreenshot()
        self.autoreset = envargs.autoreset
        self.actiondic = {
            0:"w",
            1:"a",
            2:"s",
            3:"d",
            4:"space",
            5:"shift",
        }
        try:
            self._init_windows()
        except TypeError as e:
            raise "Please start game firstly"
        self._env_record_state()
        self.reset()
    def reset(self):
        if self.autoreset:
            # Return to appointed Place
            if not self.keyboardobj._suskeyboard.is_alive():
                # self.keyboardobj.stop()
                try:
                    self.keyboardobj.start()
                except:
                    raise "SubProcess starts faild."
            # move to assigned location

        pointer_dir, target_dir, distance, img_binary = self.keyboardobj.snapandanalsis()
        rewards = self.return_rewards_logit(distance)
        img_binary = cv2.resize(img_binary,(640,640))
        img_tensor = torch.Tensor(img_binary).unsqueeze(0)
        img_tensor = torchvision.transforms.Normalize(0.5,0.5)(img_tensor)
        img_tensor = img_tensor.view(-1)
        infos = torch.tensor([pointer_dir/12, target_dir/12, distance/90], dtype=torch.float32)
        return infos, img_tensor, 0.0, False
    def step(self, act):
        action = self.actiondic.get(act)
        pointer_dir, target_dir, distance, img_binary = self.keyboardobj.execute_keyboard_down(action)
        # state = pointer_dir, target_dir, distance

        rewards = self.return_rewards_logit(distance, pointer_dir, target_dir)
        img_binary=cv2.resize(img_binary,(640,640))
        img_tensor = torch.Tensor(img_binary).unsqueeze(0)
        img_tensor = torchvision.transforms.Normalize(0.5,0.5)(img_tensor)
        img_tensor = img_tensor.view(-1)
        infos = torch.tensor([pointer_dir/12, target_dir/12, distance/90], dtype=torch.float32)

        if distance == 0 and pointer_dir ==0 and target_dir ==0:
            terminal =False
            rewards = 0.4
        else:
            if distance <3:
                terminal = True
                rewards = rewards *10

            else:
                terminal = False


        return infos, img_tensor, rewards, terminal
    def stop(self):
        keyboard.unhook_all()
        self.keyboardobj.stop()

    def return_rewards_logit(self, distance:float, pointer_d:int, tar_d:int):
        if self.ssum<50:

            self.distance_list.append({"distance":distance,"pointer":pointer_d, "target":tar_d}.copy())
            self.ssum+=1
            return 0.4
        if self.ssum == 50:

            self.records = pd.DataFrame(self.distance_list)
            self.distance = self.records['distance'].values
            self.pointer = self.records['pointer'].values
            self.target = self.records['target'].values
            del self.distance_list
            return 1.0
        avg_dist = np.mean(self.distance)
        if distance >avg_dist:
            return 0
        # print(age_column)
        # list(distance_co
        value = sum(self.distance_list)/self.ssum
        self.ssum +=1
        rewards = 1.0 if value> distance else -1.0
        return rewards
    def _init_windows(self,):
        windows_title = "原神"
        hwnd =  win32gui.FindWindow(None, windows_title)
        if hwnd !=0:
            win32gui.SetForegroundWindow(hwnd)
        else:
            raise f"Please start Game:{windows_title}"
        keyboard.on_press(on_press_event)
    def _env_record_state(self):
        # 距离
        self.ssum = 0
        self.distance_list = []
def on_press_event(event):
    print(f'Key {event.name}')
    pass
def ob_envs(envargs: EnvArgs) -> Union[AbcEnv, Env]:
    if envargs.EnvName in ENVIRONMENT_NAMES:
        envs = None
        if envargs.EnvName.startswith("Genshin"):
            try:
                envs = GenshinGames(envargs)
            except:
                print( f"Please start Game:{envargs.EnvName}")
        # print(type(envs))
        # assert envs is not  None,"Error Game haven't been init"

        return envs
    try:
        envs = gym.make(envargs.gym_id, render_mode = "human")
        return envs
    except:
        raise "Gym environment haven't built successfully"
if __name__ == "__main__":
    envargs= EnvArgs()
    env = ob_envs(envargs)

    keyboard.on_press(on_press_event)
    if env is None:
        exit()
    env.reset()
    for i in ['w','a','s','d',"space",'a','s','shift','s','s']:
        status, img, terminal = env.step(i)
        print(status.shape,img.shape)
    keyboard.unhook_all()
