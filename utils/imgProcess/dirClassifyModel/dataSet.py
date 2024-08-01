import os
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # 训练集路径
        self.root_dir = root_dir
        # 图像要进行的tensor变换--resize/toTensor等
        self.transform = transform
        # 所有图像的路径和标签存储在列表中
        self.image_paths = []
        self.labels = []
        # 进行提取路径、标签并存储的操作
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                for img_name in os.listdir(folder_path):
                    img_path = os.path.join(folder_path, img_name)
                    img_name = img_name.split('.')[0]
                    label = int(img_name.split('_')[-1])
                    if label == 12:
                        label = 0
                    self.image_paths.append(img_path)
                    self.labels.append(label)

    # 返回数据集大小
    def __len__(self):
        return len(self.image_paths)

    # 获取数据集元素
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


'''
# 使用TensorBoard可视化训练集和验证集

# 初始化TensorBoard
writer = SummaryWriter('../logs/image_classification_experiment')

# 记录一些样本图像
dataiter = iter(train_loader)
images, labels = next(dataiter)

img_grid = torchvision.utils.make_grid(images)
writer.add_image('train_images', img_grid)

# 记录训练和验证数据的大小
writer.add_text('Dataset Information', f'Training size: {train_size}\nValidation size: {val_size}')

# 关闭TensorBoard
writer.close()
'''