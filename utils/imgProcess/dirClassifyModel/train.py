import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

from dirClassifyModel.dataSet import ImageDataset

'''使用预训练的ResNet模型'''
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 12)  # 假设有12个分类

'''加载数据集'''
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

dataset = ImageDataset(root_dir='../dataset/train', transform=transform)

# 划分训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

'''定义损失函数和优化器'''
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 记录到TensorBoard
writer = SummaryWriter('runs/image_classification_experiment')

# Early stopping参数
early_stopping_patience = 10
early_stopping_counter = 0
best_val_loss = np.inf
best_model_wts = None


num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if i % 10 == 9:  # 每10个批次记录一次
            print(f'epoch:{epoch},batch_num:{i}|180,training loss:{running_loss}')
            writer.add_scalar('training loss', running_loss / 10, epoch * len(train_loader) + i)
            running_loss = 0.0

    train_acc = 100. * correct / total
    writer.add_scalar('training accuracy', train_acc, epoch)

    # 验证过程
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100. * correct / total
    writer.add_scalar('validation loss', val_loss, epoch)
    writer.add_scalar('validation accuracy', val_acc, epoch)

    print(f'Epoch {epoch + 1}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Val Loss: {val_loss:.4f}')

    # 每5轮保存一次模型
    if epoch % 5 == 0:
        model_weight = model.state_dict()
        torch.save(model_weight, f'models/{epoch}.pth')

    # 检查是否保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = model.state_dict()
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1

    # 检查是否需要早停
    if early_stopping_counter >= early_stopping_patience:
        print('Early stopping')
        break

# 保存最佳模型
torch.save(best_model_wts, 'models/best_model.pth')
print('Finished Training')

# 关闭TensorBoard
writer.close()