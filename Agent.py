import sys, os
from pathlib import Path
this_dir = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(str(Path(this_dir).parent))
import torch.nn as nn
import math

from hyparam import Modelargs

class ActionLearning(nn.Module):
    def __init__(self, modelargs: Modelargs):
        super().__init__()
        '''
            Policy Actor
            Value Function Actor : Can help us need not to get the target pennolize position for estimating the value of doing this action
        '''
        
        self.model = nn.ModuleDict(
            dict(
                stateemb = nn.Linear(modelargs.state_dim, modelargs.hidden_dim),
                
                relu = nn.ReLU(),
                layernormal = nn.LayerNorm(modelargs.hidden_dim),
                # mlp = nn.Sequential(
                #     nn.Linear(modelargs.hidden_dim, modelargs.hidden_dim),
                #     nn.Sigmoid(),
                #     nn.Linear(modelargs.hidden_dim, modelargs.hidden_dim)
                # ),
                mlp = nn.Sequential(
                    nn.Linear(modelargs.hidden_dim, modelargs.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(modelargs.hidden_dim, modelargs.hidden_dim)
                )
            )
        )
        self.model.stateemb._initmethod = True
    
        self.lm =nn.Sequential(
            nn.Linear(modelargs.hidden_dim, modelargs.action_dim),
            nn.Softmax(dim=-1)
            )
        self.lm._initmethodforaction = 4
            
        self.action_dim = modelargs.action_dim
        self.hidden_dim = modelargs.hidden_dim
        self.state_dim = modelargs.state_dim
        
        # self.apply(self._init_weight)
    
    def LoRALayer(self, ):
        pass
    def forward(self, state):
        output = self.model.stateemb(state)
        output = self.model.relu(output)
        output = self.model.layernormal(output)
        # logits = self.model.mlp(output)
        logits = self.model.mlp(output)
        logits = self.lm(logits)
        return logits
class CriticLearning(nn.Module):
    def __init__(self, modelargs: Modelargs):
        super().__init__()
        self.model = nn.ModuleDict(
            dict(
                
                actemb = nn.Embedding(modelargs.action_dim, modelargs.hidden_dim),
                
                stateemb = nn.Linear(modelargs.state_dim, modelargs.hidden_dim),
                
                relu = nn.ReLU(),
                layernormal = nn.LayerNorm(modelargs.hidden_dim),
                # mlp = nn.Sequential(
                #     nn.Linear(modelargs.hidden_dim, modelargs.hidden_dim),
                #     nn.Sigmoid(),
                #     nn.Linear(modelargs.hidden_dim, modelargs.hidden_dim)
                # ),
                mlp = nn.Sequential(
                    nn.Linear(modelargs.hidden_dim, modelargs.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(modelargs.hidden_dim, modelargs.hidden_dim)
                )
            )
        )
        self.model.stateemb._initmethod = True
    
        self.lm =nn.Linear(modelargs.hidden_dim, 1)
        self.lm._initmethodforaction = 1
        self.action_dim = modelargs.action_dim
        self.hidden_dim = modelargs.hidden_dim
        self.state_dim = modelargs.state_dim
    
    def forward(self, state, action):
        action = self.model.actemb(action).squeeze(-2)

        output = action + self.model.stateemb(state)
        output = self.model.relu(output)
        output = self.model.layernormal(output)
        # logits = self.model.mlp(output)
        logits = self.model.mlp(output)
        logits = self.lm(logits)
        return logits
class PPOAgent(nn.Module):
    def __init__(self, modelargs: Modelargs):
        super().__init__()
        self.policy = ActionLearning(modelargs)
        
        self.valueExp = CriticLearning(modelargs)
        self.std = 1 /math.sqrt(modelargs.hidden_dim)
        self.action_dim = modelargs.action_dim
        self.apply(self._init_weight)
        
    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=1/math.sqrt(self.action_dim))
            if hasattr(module, "_initmethod"):
                nn.init.normal_(module.weight, mean=0.0, std = 1/math.sqrt(self.action_dim))
            if hasattr(module, "initmethodforaction"):
                nn.init.normal_(module.weight, mean=0.0, std = 1/math.sqrt(self.std))
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std = 1/math.sqrt(self.std))   
    def forward(self,):
        raise NotImplementedError
    def act(self, state):
        action = self.policy(state)
        return action
        
    def critic(self, state, action):
        
        return self.valueExp(state, action)