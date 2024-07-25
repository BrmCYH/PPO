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
import threading
from typing import Literal

ENVIRONMENT_NAMES= ["Genshin_Impact"]
class AbcEnv(ABC):
    def step(action):
        '''
            Return the environment callback based on action
        '''
    def reset():
        """
            Reset the environments
        """
class Games(AbcEnv):
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
        self.reset()
    def reset(self):
        if self.autoreset:
            # Return to appointed Place
            self.keyboardobj.stop()
            self.keyboardobj.start()
            pass
        try:
            self.keyboardobj.start()
        except:
            pass
        pass
    async def step(self, action):
        # act = action.item()# action 
        return await self.keyboardobj.execute_keyboard_down(action)
        # self.keyboardobj.execute_keyboard_down(act=act)
    def keyfeed(self):
        pass
    def capture_feedback(self):
        
        for self.screenshot_generator, self.keyboard_feedback in self.take_screenshot("step"), self.keyfeed():
            
        
            if self.exit_event.is_set():
                break
    def take_screenshot(self, status:Literal["step","reset"]):
        if status == "reset":
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                screenshot = sct.grab(monitor)
                screenshot_np = np.array(screenshot)
                return screenshot_np
        with mss.mss() as sct:
            while self.exit_event.is_set():
                
                monitor = sct.monitors[1]
                screenshot = sct.grab(monitor)
                screenshot_np = np.array(screenshot)
                yield screenshot_np
def ob_envs(envargs: EnvArgs) -> Union[AbcEnv, Env]:
    if envargs.EnvName in ENVIRONMENT_NAMES:
        raise "Game Environment have not been builded now"
    try:
        envs = gym.make(envargs.gym_id, render_mode = "human")
        return envs
    except:
        raise "Gym environment haven't built successfully"