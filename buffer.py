import torch
import numpy as np
from typing import List, Tuple, Dict, Literal, Union
class MemoryBuffer:
    def __init__(self, maxsize, statedim: int, actdim: int, device:Union[str, torch.device]):
        
        self.maxsize = int(maxsize)
        self.statedim = statedim
        self.actdim = actdim
        
        self.return_tensor = True
        self.device = device if torch.cuda.is_available() else 'cpu'
    @property
    def _curr_size(self):
        try:
            return self._state.shape[0]
        except:
            return 0
    def _init_buffer(self, kwargs):

        self._state = kwargs['obs']
        self._img = kwargs['img']
        self._action = kwargs['act']
        self._reward = kwargs['reward']
        self._next_state = kwargs['next_obs']
        self._next_img = kwargs['next_img']
        self._logprob = kwargs['logprob']
        self._terminal = kwargs['terminal']
        self._state_values = kwargs['values']


    def append(self, obs, img, act, reward, next_obs, next_img, logprob, terminal, values):
        snap = self.align_tensor(self.return_tensor, obs = obs, img= img, act = act, reward = reward, next_obs= next_obs, next_img=next_img, logprob=logprob, terminal=terminal, values=values)
        
        if self._curr_size >= self.maxsize:
            # import code; code.interact(local=locals())
            self._state      =  torch.cat((snap['obs'],self._state), dim = 0)[:-1, :]
            self._img        =  torch.cat((snap["img"],self._img)  , dim = 0)[:-1,:]
            self._action     =  torch.cat((snap['act'],self._action), dim = 0)[:-1, :]
            self._reward     =  torch.cat((snap['reward'], self._reward), dim = 0)[:-1, :]
            self._next_state =  torch.cat((snap['next_obs'],self._next_state, ), dim = 0)[:-1, :]
            self._next_img =  torch.cat((snap['next_img'],self._next_img, ), dim = 0)[:-1, :]
            self._logprob    =  torch.cat((snap['logprob'],self._logprob), dim = 0)[:-1, :]
            self._terminal   =  torch.cat((snap['terminal'],self._terminal), dim = 0)[:-1, :]
            self._state_values= torch.cat((snap['values'],self._state_values), dim = 0)[:-1, :]
        else:
            if self._curr_size == 0:
                self._init_buffer(snap)
            else:

                self._state      =  torch.cat((snap['obs'],self._state), dim = 0)
                self._img        =  torch.cat((snap['img'],self._img), dim = 0)
                self._action     =  torch.cat((snap['act'],self._action), dim = 0)
                self._reward     =  torch.cat((snap['reward'],self._reward), dim = 0)
                self._next_state =  torch.cat((snap['next_obs'],self._next_state), dim = 0)
                self._next_img =  torch.cat((snap['next_img'],self._next_img), dim = 0)
                self._logprob    =  torch.cat((snap['logprob'],self._logprob), dim = 0)
                self._terminal   =  torch.cat((snap['terminal'],self._terminal), dim = 0)
                self._state_values= torch.cat((snap['values'],self._state_values), dim = 0)
           
            
    @staticmethod
    def type_fine(it):
    
        if not hasattr(it, "ndim"):
            if isinstance(it, bool):
                
                x = torch.tensor(it, dtype = torch.bool).unsqueeze(0).unsqueeze(0)
            elif isinstance(it, float):
                x = torch.tensor(it, dtype= torch.float32).unsqueeze(0).unsqueeze(0)
            else: 
                x = torch.tensor(it, dtype= torch.int32).unsqueeze(0).unsqueeze(0)
        else:
            if it.ndim == 0:
                if isinstance(it, np.int64):
                    x = torch.tensor(it, dtype=torch.int32).unsqueeze(0).unsqueeze(0)
                else:
                    x = torch.tensor(it, dtype= torch.float32).unsqueeze(0).unsqueeze(0)

            elif it.ndim == 1:
                
                if isinstance(it, np.ndarray):
                    x = torch.Tensor(it).unsqueeze(0)
                else:
                    x = it.clone().detach().requires_grad_(True).unsqueeze(0)
            else:
                x = it.clone().detach().requires_grad_(True)
        assert x.dim()==2, 'the dim of {it} is not compaired with standard'
        return x
    def align_tensor(self, return_tensor, **kwargs) -> Tuple[List[torch.Tensor]]:
        if return_tensor:
            buff = {}
            for k, v in kwargs.items():
                if k =='obs':
                    if self._curr_size >0:
                        assert v.shape[-1] == self._state.shape[-1], f"{k} is not compair with buffer size"
                    else:
                        assert v.shape[-1] == self.statedim,f"{k} is not compair with buffer size"
                    buff[k] = MemoryBuffer.type_fine(v).to(self.device)
                elif k =='img':
                    buff[k] = MemoryBuffer.type_fine(v).to(self.device)
                elif k =='act':
                    buff[k] = MemoryBuffer.type_fine(v).to(self.device)
                elif k == 'reward':
                    buff[k] = MemoryBuffer.type_fine(v).to(self.device)
                elif k == 'next_obs':
                    if self._curr_size >0:
                        assert v.shape[-1] == self._state.shape[-1], f"{k} is not compair with buffer size"
                    else:
                        assert v.shape[-1] == self.statedim,f"{k} is not compair with buffer size"
                    buff[k] = MemoryBuffer.type_fine(v).to(self.device)
                elif k == 'next_img':
                    buff[k] = MemoryBuffer.type_fine(v).to(self.device)
                elif k == 'logprob':
                    buff[k] = MemoryBuffer.type_fine(v).to(self.device)
                elif k == 'values':
                    buff[k] = MemoryBuffer.type_fine(v).to(self.device)
                elif k == 'terminal':
                    buff[k] = MemoryBuffer.type_fine(v).to(self.device)
                else:
                    raise f"Input item error:detail {k} "
            
        return buff
    def get_r_trend(self, compare_include:Literal['average','pioneer','expert','increment'], gamma:float=0.1)-> torch.Tensor:
        '''
            How to consider the feedback from envs with rewards?   
            According to varition from rewards, get Advantage from buffer
            gamma: decay radio , to build ties with front rewards
                
        '''
        if compare_include == 'average':
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(self._reward.contiguous().view(-1), self._terminal.contiguous().view(-1)):
                
                '''
                    reversed the buffer  
                    and decay the weight with deep
                '''
                # 奖励路径
                if is_terminal.item():
                    discounted_reward=0
                
                discounted_reward = reward + (gamma * discounted_reward) 
                rewards.insert(0, discounted_reward) 
            rewards = torch.tensor(rewards, dtype= torch.float32)
            rewards= (rewards - rewards.mean()) / (rewards.std() + 1e-7)
            return rewards
        elif compare_include == 'increment':
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(self._reward.contiguous().view(-1), self._terminal.contiguous().view(-1)):
                
                if is_terminal.item():
                    discounted_reward=0
                
                discounted_reward = reward + (gamma * discounted_reward) 
                rewards.insert(0, discounted_reward) 
            rewards = torch.tensor(rewards, dtype= torch.float32)
            rewards= (rewards - rewards.mean()) / (rewards.std() + 1e-7)
            rewards_mask = rewards<0
            rewards.masked_fill(rewards_mask,float(1e-7))
            return rewards
        else:
            raise f'{compare_include} is not be supported now'
    def clear(self):
        # self._curr_size = 0
        self._state =None
        self._img = None
        self._next_state=None
        self._next_img =None
        self._reward = None
        self._terminal = None
        self._logprob =None
        self._action=None
        self._state_values =None
    def return_tensors(self) -> Tuple[torch.Tensor]:
        '''
            Return buffer storage, old_state, reward, action, 
            state, log prob tensor, terminal tensor[torch.bool],
            state_value[value Agent predict]
        '''
        return self._state, self._img, self._reward, self._action, self._next_state, self._next_img, self._logprob, self._state_values
        
    def size(self):
        return self._curr_size
    def __len__(self):
        return self._curr_size
    def update(self):
        return self._curr_size >= self.maxsize
    def __name__(self):
        return "PPO update buffer"
    def __repr__(self) -> str:
        name = self.__name__()
        if self._curr_size == 0:
            
            return f'Buffer object:{name} is Empty Now!'
        
        return f"Buffer object:{name}; currently used exausted:{self._curr_size}/{self.maxsize}"
    def __show__(self,)-> Dict[str,torch.Tensor]:
        infos = dict(
            detail = dict(
                obs = self._state,
                img = self._img,
                reward = self._reward,
                action = self._action,
                next_obs = self._next_state,
                next_img = self._next_img,
                logprob = self._logprob,
                terminal = self._terminal,
                values = self._state_values
        ),
            shape = dict(
                obs = self._state.shape,
                img = self._img.shape,
                reward = self._reward.shape,
                action = self._action.shape,
                next_obs = self._next_state.shape,
                next_img = self._next_img.shape,
                logprob = self._logprob.shape,
                terminal = self._terminal.shape,
                values = self._state_values.shape
            )
        )
        return infos
