import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from skorch import NeuralNetClassifier
from torch.utils.data import random_split
import torch.optim as optim

#   Hyper parameters
NUM_EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.001
IMG_SIZE = 32

# Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#  Get and split dataset
dataset = torchvision.datasets.ImageFolder(root = './data', transform=transform)
train_dataset, test_dataset = train_test_split(dataset, test_size = 0.2)

print(len(test_dataset))

# Load data

train_loader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE, shuffle = False)

classes = ('maskless_mask','cloth_mask', 'surgical_mask', 'n95_mask')
NUM_CLASSES = len(classes)

"""
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(BATCH_SIZE)))
"""

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

m = len(train_dataset)
val_size= int(m * 0.2)
train_size = (m - int(m * 0.2))
train_data , val_data =random_split(train_dataset, [train_size,val_size])

y_train = np.array([y for x, y in iter(train_data)])

torch.manual_seed(0)
net = NeuralNetClassifier(
    ConvNN,
    max_epochs=2,
    iterator_train__num_workers=0,
    iterator_valid__num_workers=0,
    lr=1e-3,
    batch_size=32,
    optimizer=optim.SGD,
    criterion=nn.CrossEntropyLoss
)
net.fit(train_data, y=y_train)
y_pred = net.predict(test_dataset)
y_test = np.array([y for x, y in iter(test_dataset)]) 

print(f"Precision is: {100 * precision_score(y_test,y_pred, average='weighted'):.3f} %")
print(f"Accuracy is: {100 * accuracy_score(y_test,y_pred):.3f} %")
print(f"Recall is: {100 * recall_score(y_test,y_pred, average='weighted'):.3f} %")
print(f"F1 is: {100 * f1_score(y_test,y_pred, average='weighted'):.3f} %")

plot_confusion_matrix(net, test_dataset, y_test.reshape(-1, 1),display_labels= classes)
plt.show()

