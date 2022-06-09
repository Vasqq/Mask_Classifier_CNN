import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Device config

#device = 'cpu'
#device = 'cuda'
#print(f'Is GPU avaialble?\n {torch.cuda.is_available()}')

#   Hyper parameters
NUM_EPOCHS = 100
BATCH_SIZE = 5
LEARNING_RATE = 0.001
IMG_SIZE = 32

# Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#  Get data

train_dataset = torchvision.datasets.ImageFolder(root = './data/training', transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root = './data/test', transform=transform)

# Load data

train_loader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = False)

classes = ('maskless_mask','cloth_mask', 'surgical_mask', 'n95_mask')
NUM_CLASSES = len(classes)


# Define model

class ConvNN(nn.Module):
    def __init__(self):
        super(ConvNN, self).__init__()
        self.conv_layer = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
        nn.Dropout(p=0.1),
        nn.Linear(8 * 8 * 64, 1000),
        nn.ReLU(inplace=True),
        nn.Linear(1000, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.1),
        nn.Linear(512, 10)
        )

    def forward(self, x):
        # conv layers
        x = self.conv_layer(x)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = self.fc_layer(x)
        return x

model = ConvNN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE)

n_total_steps = len(train_loader)
loss_list = []
acc_list = []
"""

#   Training loop

for epoch in range(NUM_EPOCHS):

    for i, (images,labels) in enumerate(train_loader):

        #   Forward pass
        outputs = model(images)
        loss = criterion(outputs,labels)
        loss_list.append(loss.item())

        #   Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(i+1) % 12 == 0:
            print(f'epoch {epoch + 1} / {NUM_EPOCHS}, step {i + 1}/{n_total_steps}, loss = {loss.item():.4f}')

"""
#FILE = "trained_model.pth"
#torch.save(model.state_dict(), FILE)
PATH = './trained_model.pth'
model = ConvNN()
model.load_state_dict(torch.load(PATH))

# Testing

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(NUM_CLASSES)]
    n_class_samples = [0 for i in range(NUM_CLASSES)]

    for images, labels in test_loader:
        outputs = model(images)

        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(BATCH_SIZE):
            label = labels[i]
            pred = predicted[i]

            if (label == pred):
                n_class_correct[label] +=1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct/n_samples
    print('\n***TEST DATA***')
    print(f'\nOverall accuracy = {acc:.3f} %\n')

    for i in range(NUM_CLASSES):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc:.3f} %')
