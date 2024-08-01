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
import gc
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
        self.device = kwargs['device'] if hasattr(kwargs,'device') else 'cpu'
        self.max = trainingargs.epochs
        self.clip_value = trainingargs.grad_clip
        # if torch.cuda.is_available() and self.device == 'cpu':
        #     self.device = 'cuda'
        # self.pioneer.to(device=self.device)
        if trainingargs.test or trainingargs.speed:
            try:
                self.pionner = torch.compile(self.pioneer)
            except:
                print("Haven't init for compile")
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
            
            coeff = 0.8 * (math.cos(math.pi * steps/self.K_epochs))
            return self.policy_lr * (1+ coeff * decay_ratio)
        else:
            decay_ratio = steps/(self.K_epochs*self.max)
            assert 0 <= decay_ratio <= 1
            
            coeff = 0.8 * (math.cos(math.pi * steps/self.K_epochs))
            return self.critic_lr * (1+ coeff * decay_ratio)
        

    def select_action(self, state: Union[numpy.ndarray, torch.Tensor], img):
        state = MemoryBuffer.type_fine(state)
        img = MemoryBuffer.type_fine(img)
        # state = state.to(self.device)
        # img = img.to(self.device)
        action, action_logprob, state_val = self.policy_act(state, img)
        
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
        old_states, old_img, real_rewards, old_actions, _ ,_ , old_logprobs, old_state_values = self.buffer.return_tensors()
        
        advantages = self.get_advantage(rewards, old_state_values)
        
        for steps in range(1, self.K_epochs+1): 
            logprobs, state_values, dist_entropy = self.pioneer_act(old_states, old_img, old_actions)
            
            state_values = torch.squeeze(state_values, dim=0)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1- self.eps_clip, 1+ self.eps_clip).mean()
            self.optimizer.zero_grad()
            # print(state_values.shape, real_rewards.shape)
            loss2 = self.Mseloss(state_values, real_rewards).mean()
            loss = -torch.min(surr1, surr2) + 0.5 * loss2 - 0.01 * dist_entropy# loss = -min(exp(对数概率密度), tensor) + 0.5* Mseloss(value, reward)
            loss = loss.mean()

            loss.backward()
            print("loss",loss)
            norm = torch.nn.utils.clip_grad_norm_(self.pioneer.parameters(), self.clip_value)
            
            for ith, param_group in enumerate(self.optimizer.param_groups):
                lr = self.get_lr(steps= steps,ith=ith)
                param_group['lr'] = lr
            
            self.optimizer.step()
        self.buffer.clear()
        gc.collect()
        return loss.mean().item(), lr, norm
    def topn(self, action:torch.Tensor):
        
        return action.argmax(-1).item()

    def policy_act(self, state, img):
        # state=state.unsqueeze(0)
        # state = state.to(self.device)
        # img = img.to(self.device)
        with torch.no_grad():

            action = self.pioneer.act(state, img)
            dist = Categorical(action)
            action = dist.sample()

            action_logprob = dist.log_prob(action)
            act = action.unsqueeze(0)
            # act = act.to(self.device)
            state_val = self.pioneer.critic(state, img,  act)
        return action.detach(), action_logprob, state_val
    
    def pioneer_act(self, state, img, action):
        # state = state.to(self.device)
        # img = img.to(self.device)
        # action = action.to(self.device)
        pio_action = self.pioneer.act(state, img)
        dist = Categorical(pio_action)
        action_logprob = dist.log_prob(action)
        dist_entropy = dist.entropy() 
        state_values = self.pioneer.critic(state, img, action)
        return action_logprob, state_values, dist_entropy
