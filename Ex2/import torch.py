import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from torch.utils.data import random_split
import os
import time
import copy

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="The \"freeze_support\" line can be omitted...")

# Set data directory
data_dir = r'E:\flower_dataset'

# Data augmentation and normalization for training and validation
data_transforms = transforms.Compose([
        # GRADED FUNCTION: Add five data augmentation methods, Normalizating and Tranform to tensor
        ### START SOLUTION HERE ###
        # Add five data augmentation methods, Normalizating and Tranform to tensor
        ### END SOLUTION HERE ###
    transforms.RandomResizedCrop(224),          # 随机裁剪并调整大小
    transforms.RandomHorizontalFlip(),          # 随机水平翻转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
    transforms.RandomRotation(30),              # 随机旋转 ±30 度
    transforms.ToTensor(),                      # 转换为 Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# Load the entire dataset
full_dataset = datasets.ImageFolder(data_dir, data_transforms)

# Automatically split into 80% train and 20% validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Use DataLoader for both train and validation datasets
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

dataloaders = {'train': train_loader, 'val': val_loader}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

# Get class names from the dataset
class_names = full_dataset.classes

# Load pre-trained model and modify the last layer
model = models.resnet18(pretrained=True)


# GRADED FUNCTION: Modify the last fully connected layer of model
### START SOLUTION HERE ###
# Modify the last fully connected layer of model
### END SOLUTION HERE ###
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)  # 替换为5分类的全连接层


# GRADED FUNCTION: Define the loss function
### START SOLUTION HERE ###
# Define the loss function
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
### END SOLUTION HERE ###


# GRADED FUNCTION: Define the optimizer
### START SOLUTION HERE ###
# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器
### END SOLUTION HERE ###

# Learning rate scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training function
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Print learning rate for current epoch
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Learning Rate: {current_lr:.6f}')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        
                        # GRADED FUNCTION: Backward pass and optimization
                        ### START SOLUTION HERE ###
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        optimizer.zero_grad()  # 清空梯度
                        loss.backward()        # 反向传播
                        optimizer.step()       # 更新参数
                        # Backward pass and optimization
                        ### END SOLUTION HERE ###

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()  # Update learning rate based on scheduler

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Save the model if validation accuracy is the best so far
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # Save the best model
                save_dir = 'Ex2/work_dir'
                os.makedirs(save_dir, exist_ok=True)

               # GRADED FUNCTION: Save the best model
                ### START SOLUTION HERE ###
                torch.save({
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'best_acc': best_acc,
                            }, os.path.join(save_dir, 'best_model.pth'))
                # Save the best model
                ### END SOLUTION HERE ###

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model

# Train the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model = train_model(model, criterion, optimizer, scheduler, num_epochs=25)