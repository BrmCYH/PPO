import gymnasium as gym
import mss
import cv2
import time
from gymnasium import Env
from hyparam import EnvArgs
from typing import Union
from abc import ABC
import numpy as np
from utils import Interface, KeyboardScreenshotThread
import torch
from typing import Literal
import win32gui
import torchvision
import numpy
from numpy import ndarray
transformers = torchvision.transforms.Compose(
    torchvision.transforms.ToTensor(),

    
)

ENVIRONMENT_NAMES= ["Genshin_Impact"]
REGISTER_ENVS={}
KEY_VDIC = {
    
}
class AbcEnv(ABC):
    def step(action):
        '''
            Return the environment callback based on action
        '''
    def reset():
        """
            Reset the environments
        """
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
        self.keyboardobj = KeyboardScreenshotThread()
        self.autoreset = envargs.autoreset
        self._init_windows()
        self._env_record_state()
        self.reset()
    def reset(self):
        if self.keyboardobj._suskeyboard.is_alive():
            
            self.keyboardobj.stop()
        else:
            return self.step
    def step(self, action):
        
        pointer_dir, target_dir, distance, img_binary = self.keyboardobj.execute_keyboard_down(action)
        # state = pointer_dir, target_dir, distance
        rewards = self.return_rewards_logit(distance)
        pointer, target, distance= pointer_dir/12, target_dir/12, distance/90
        state = torch.tensor([pointer, target, distance],dtype=torch.float32)
        view_img = torch.Tensor(img_binary).view(-1)
        state = torch.cat([state, view_img],dim = -1)
        return state, rewards, False
    def return_rewards_logit(self, distance:float):
        if self.ssum<50:
            self.distance_list.append(distance)
            self.ssum+=1
            return 0
        if not self.ssum %20:
            value = sum(self.distance_list)/self.ssum
            self.ssum +=1
        rewards = 1 if value> distance else -1
        return rewards
        
            
    def _env_record_state(self):
        # 距离
        self.ssum = 0
        self.distance_list = []
        
    def _init_windows(self, windows_title):
        windows_title = "原神"
        
        hwnd =  win32gui.FindWindow(None, windows_title)
        if hwnd !=0:
            win32gui.SetForegroundWindow(hwnd)
        else:
            raise f"Please start Game:{windows_title}"

def ob_envs(envargs: EnvArgs) -> Union[AbcEnv, Env]:
    if envargs.EnvName in ENVIRONMENT_NAMES:
        if envargs.EnvName.startswith("Genshin"):
            
            envs = GenshinGames(envargs)
        return envs
    try:
        envs = gym.make(envargs.gym_id, render_mode = "human")
        return envs
    except:
        raise "Gym environment haven't built successfully"

    