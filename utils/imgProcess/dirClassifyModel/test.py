import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn

from dirClassifyModel.dataSet import ImageDataset


def evaluate_model(model, dataloader, criterion):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    average_loss = running_loss / len(dataloader)
    return accuracy, average_loss

def main():
    test_dir = '../dataset/test'
    model_path = './models/best_model.pth'

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    test_dataset = ImageDataset(root_dir=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 12)
    model.load_state_dict(torch.load(model_path))
    criterion = nn.CrossEntropyLoss()

    test_accuracy, test_loss = evaluate_model(model, test_loader, criterion)

    print(f'Test Accuracy: {test_accuracy:.2f}%, Test Loss: {test_loss:.4f}')


if __name__ == '__main__':
    main()
