import os

import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn

# 加载模型
model_path = 'D:\\manipulat\\aitesting-findpath-main\\utils\\imgProcess\\dirClassifyModel\\models\\best_model.pth'
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 12)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def getPointerDir(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    if transform:
        image = transform(image).unsqueeze(0)
    output = model(image)
    _, predicted_class = torch.max(output, 1)
    return predicted_class.numpy()[0]

