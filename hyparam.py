from dataclasses import dataclass, field
from typing import Union, Literal, Optional
@dataclass
class Trainingargs:
    max_steps:int=32
    gamma:float =0.001
    eps_clip:float =0.1
    test:bool = False
    speed:bool = True
    basepath: str = None
    
    lr_policy: float = 0.005
    lr_critic: float = 0.003
    epochs: int =200
    buffersize: int = 64
    episodes: int = 500
    grad_clip: float = 1.0
    cache_dir: Optional[str]=None
@dataclass 
class Modelargs:
    state_dim: int
    action_dim: int
    hidden_dim: int
    load_from_disk :Optional[str] = None
@dataclass
class EnvArgs:
    EnvName: Optional[str] = field(
        default="",
        metadata={
            "help":(
                "Gym env projects or Local games"
            )
        }
    )
    gym_id: Optional[str] = field(
        default="LunarLander-v2",
        metadata={
            "help":"custom Game Index"
        }
    )
    max_episode_steps: Optional[str] = field(
        default=500,
        metadata={
            "help":"The number of iteration in each training task."
        }
    )
    autoreset: Optional[bool] = field(
        default=True,
        metadata={
            "help":""
        }
    )
    apply_api_compatibility: Optional[bool] = field(
        default=True,
        metadata={
            "help":(
                "Whether Using termination and truncation to replace Done or not!",
                "Termination and truncation incloud more information than Done ."
            )
        }
    )