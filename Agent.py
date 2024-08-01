import sys, os
from pathlib import Path
this_dir = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(str(Path(this_dir).parent))
import torch.nn as nn
import math
import torch.nn.functional as F
from hyparam import Modelargs
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        # 深度可分离卷积层包含深度卷积和逐点卷积两部分
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # 加入批量归一化
        self.pointwise_bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = self.depthwise(x)
        x = F.relu(x)
        x = self.pointwise(x)
        x = F.relu(x)
        x = self.pointwise_bn(x)
        return x

# MobileNet-like 特征提取器
class MobileNetLikeFeatureExtractor(nn.Module):
    def __init__(self, input_shape, hidden_dim):
        super(MobileNetLikeFeatureExtractor, self).__init__()
        self.input_shape = [input_shape]*2
        channels = 1  # 输入图像是单通道
        # 320
        self.initial_conv = nn.Conv2d(channels, 16, kernel_size=3, stride=2, padding=1)
        # 160
        self.dw_conv1 = DepthwiseSeparableConv(16, 32, stride=2)
        # 80
        self.dw_conv2 = DepthwiseSeparableConv(32, 64, stride=2)
        # 40
        self.dw_conv3 = DepthwiseSeparableConv(64, 128, stride=2)
        # 20
        self.dw_conv4 = DepthwiseSeparableConv(128, 256, stride=2)
        # 20
        self.dw_conv5 = DepthwiseSeparableConv(256, hidden_dim, stride=1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        # 调用权重初始化函数
        # self._initialize_weights()

    def forward(self, x):
        # 恢复为原始图像的形状 (batch_size, channels, height, width)
        # 现在的尺寸为 (batch_size, 1, 640, 640)
        x = x.view(-1, *self.input_shape).unsqueeze(1)

        # 经过 initial_conv 和 ReLU 之后，尺寸变为 (batch_size, 16, 320, 320)
        # 初始卷积层将通道从1增加到16，步幅为2导致尺寸减半
        x = self.initial_conv(x)
        x = F.relu(x)

        x = self.dw_conv1(x)
        x = self.dw_conv2(x)
        x = self.dw_conv3(x)
        x = self.dw_conv4(x)
        x = self.dw_conv5(x)
        x = self.avg_pool(x)
        x = self.flatten(x)

        return x

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
                img_model = MobileNetLikeFeatureExtractor(modelargs.img_state_dim, modelargs.hidden_dim),
                share_weight = nn.Linear(modelargs.hidden_dim, modelargs.hidden_dim, bias=False),
                relu = nn.ReLU(),
                layernormal = nn.LayerNorm(modelargs.hidden_dim),
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


    def LoRALayer(self, ):
        pass
    def forward(self, state, img):
        output = self.model.stateemb(state)
        img_output = self.model.img_model(img)

        output = self.model.relu(output)

        output = img_output + output
        output = self.model.share_weight(output)
        output = self.model.layernormal(output)

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
                img_model = MobileNetLikeFeatureExtractor(modelargs.img_state_dim, modelargs.hidden_dim),
                share_weight = nn.Linear(modelargs.hidden_dim, modelargs.hidden_dim, bias=False),
                relu = nn.ReLU(),
                layernormal = nn.LayerNorm(modelargs.hidden_dim),
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

    def forward(self, state,img, action):
        action = self.model.actemb(action).view(-1, self.hidden_dim)

        img_output = self.model.img_model(img)
        output = self.model.stateemb(state)
        output = self.model.relu(output)
        output = img_output + output + action
        output = self.model.share_weight(output)
        output = self.model.layernormal(output)

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
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self,):
        raise NotImplementedError
    def act(self, state, img):
        action = self.policy(state, img)
        return action

    def critic(self, state, img, action):

        return self.valueExp(state, img, action)
