import sys, os
from pathlib import Path
this_dir = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(str(Path(this_dir).parent))
import numpy
import os, math
import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Union

from buffer import MemoryBuffer
from Agent import PPOAgent
from hyparam import Modelargs, Trainingargs
class PPOTrain:
    def __init__(
        self, 
        modelargs: Modelargs,
        trainingargs: Trainingargs,
        **kwargs
        ):
        self.cache_path = trainingargs.basepath+"/"+'cache_dir' if trainingargs.basepath is not None else str(os.path.join(os.getcwd(),'cache_dir'))
        try:
            os.makedirs(self.cache_path)
        except FileExistsError as e:
            print('cache dir has been created')
        self.pioneer = PPOAgent(modelargs)
        self.buffer = MemoryBuffer(trainingargs.buffersize, modelargs.state_dim, modelargs.action_dim, device="cpu")
        self.gamma = trainingargs.gamma
        self.policy_lr = trainingargs.lr_policy
        self.critic_lr = trainingargs.lr_critic
        self.optimizer = torch.optim.AdamW([
                    {'params': self.pioneer.policy.parameters(), 'lr': trainingargs.lr_policy},
                    {'params': self.pioneer.valueExp.parameters(), 'lr': trainingargs.lr_critic}
                ])
        self.K_epochs=  trainingargs.max_steps 
        self.Mseloss = nn.MSELoss() # 
        self.eps_clip = trainingargs.eps_clip
        self.device = 'cpu'
        self.max = trainingargs.epochs
        self.clip_value = trainingargs.grad_clip
        if torch.cuda.is_available():
            self.device = 'cuda'
        if trainingargs.test or trainingargs.speed:
            self.pionner = torch.compile(self.pioneer)
        if kwargs['state_dict'] is not None:
            names = kwargs['state_dict'].keys()
            names = [k for k in names if not k.startswith('policy.model.link')]
            sd_pion = self.pioneer.state_dict()
            for name in names:
                assert sd_pion[name].shape == kwargs['state_dict'][name]
                with torch.no_grad():
                    sd_pion[name].copy_(kwargs['state_dict'][name])
            self.pioneer.load_state_dict(sd_pion)
            print("weight load from pretrained model.")
    def get_lr(self, steps, ith):
        if ith ==0:
            
            decay_ratio = steps/(self.K_epochs*self.max)
            assert 0 <= decay_ratio <= 1
            
            coeff = 0.8 * (math.cos(math.pi * steps/self.K_epochs)) # coeff starts at 1 and goes to 0
            return self.policy_lr * (1+ coeff * decay_ratio)
        else:
            decay_ratio = steps/(self.K_epochs*self.max)
            assert 0 <= decay_ratio <= 1
            
            coeff = 0.8 * (math.cos(math.pi * steps/self.K_epochs)) # coeff starts at 1 and goes to 0
            return self.critic_lr * (1+ coeff * decay_ratio)
        

    def select_action(self, state: Union[numpy.ndarray, torch.Tensor]):
        state = MemoryBuffer.type_fine(state)
        action, action_logprob, state_val = self.policy_act(state)
        
        return action, action_logprob, state_val

    def infos_to_device(self):
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)
        return old_states, old_actions, old_logprobs, old_state_values
    def get_advantage(self, rewards:torch.Tensor, value_pr:torch.Tensor):
        '''
            Get advantage between reward-truth and value_pr-predict.
            return the variation radio and normalize
        '''
        advantages = rewards.detach() - value_pr.detach()
        
        return advantages
    def update(self):
        rewards = self.buffer.get_r_trend('average',gamma = self.gamma)        
        old_states, rewards, old_actions, _ , old_logprobs, old_state_values = self.buffer.return_tensors()
        
        advantages = self.get_advantage(rewards, old_state_values)
        
        for steps in range(1, self.K_epochs+1):
            # Forward with states_buffer, action_buffer  
            logprobs, state_values, dist_entropy = self.pioneer_act(old_states, old_actions)
            
            state_values = torch.squeeze(state_values, dim=0)
            # compute loss Policy Action PDF 
            # the probility of action \frac{e^{logprob}}{e^{old_logprob}} = \frac{p}{p_o}
            # 1. 采样与实际偏差降低
            ratios = torch.exp(logprobs - old_logprobs.detach()) # old_policy -> action - policy->action ,keep the variation between pionner and lazzy model smoothly.
            # 2. decline the gap between old_logprob and logprob which items in old_logprob are 
            # ratios = torch.exp(logprobs)/torch.exp(old_logprobs*torch.log(advantages))
            # print(ratios)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1- self.eps_clip, 1+ self.eps_clip).mean()# clip * adantange
            # the weight of dist_entropy have more bigger, the action from policy agent would be more random
            # -min(surr1, surr2) optimize the policy to use
            # mse to optimize reward value net
            # dist_entropy
            # minimize loss function :1. Max(surr1, surr2) 2.Min(reward-pr) 3. max entropyof action
            self.optimizer.zero_grad()
            # 0.01 ->0.09
            loss = -torch.min(surr1, surr2) + 0.5 * self.Mseloss(state_values, rewards) - 0.01 * dist_entropy# loss = -min(exp(对数概率密度), tensor) + 0.5* Mseloss(value, reward)
            # gradient accumulate
            loss = loss.mean()
            loss.backward()
            
            norm = torch.nn.utils.clip_grad_norm_(self.pioneer.parameters(), self.clip_value)
            
            for ith, param_group in enumerate(self.optimizer.param_groups):
                lr = self.get_lr(steps= steps,ith=ith)
                param_group['lr'] = lr
            
            self.optimizer.step()
        
        return loss.mean().item(), lr, norm
    def topn(self, action:torch.Tensor):
        
        return action.argmax(-1).item()
    @torch.no_grad
    def policy_act(self, state):
        state=state.unsqueeze(0)
        action = self.pioneer.act(state)
        # print(f"lazzyr {action}")
        dist = Categorical(action)
        action = dist.sample() # choose action
        
        action_logprob = dist.log_prob(action) # 对数概率密度
        act = action.unsqueeze(0)
        state_val = self.pioneer.critic(state, act)# get value
        return action.detach(), action_logprob, state_val
    
    def pioneer_act(self, state, action):
        pio_action = self.pioneer.act(state)
        # print(f"Pioneer {pio_action}")
        dist = Categorical(pio_action)
        # actions = dist.sample()
        
        # actions = pio_action.argmax(-1)
        action_logprob = dist.log_prob(action)# 对数概率密度
        dist_entropy = dist.entropy() # 信息熵 该动作的
        state_values = self.pioneer.critic(state, action)
        return action_logprob, state_values, dist_entropy