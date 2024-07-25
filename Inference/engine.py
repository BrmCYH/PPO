import sys, os
from pathlib import Path
this_dir = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(str(Path(this_dir).parent.parent))
import torch
from Agent import PPOAgent
from hyparam import Modelargs
class engine:
    def __init__(self, model_path, modelargs:Modelargs):
        
        state_dict = torch.load('C:\\Users\\happyelements\\workstation\\RLWorkstation\\cache_dir\\reward_better_version_bt.pt')
        assert state_dict['']
        self.Agent = PPOAgent(modelargs)
        self.Agent.load_state_dict(state_dict)
        
    @torch.inference_mode
    def act(self, state):
        action = self.Agent.act(state)
        
        return action.argmax(-1)