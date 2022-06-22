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
import pickle

# Hyper parameters
NUM_EPOCHS = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.001
IMG_SIZE = 32

# Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#  Get and split dataset in training and testing datasets
dataset = torchvision.datasets.ImageFolder(root = './data', transform=transform)
m = len(dataset)
test_size= int(m * 0.2)
train_size = (m - int(m * 0.2))
train_dataset , test_dataset =random_split(dataset, [train_size, test_size])

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

m = len(train_dataset)
val_size= int(m * 0.2)
train_size = (m - int(m * 0.2))
train_data , val_data =random_split(train_dataset, [train_size,val_size])

y_train = np.array([y for x, y in iter(train_data)])

torch.manual_seed(0)
net = NeuralNetClassifier(
    ConvNN,
    max_epochs=50,
    iterator_train__num_workers=0,
    iterator_valid__num_workers=0,
    lr=1e-3,
    batch_size=32,
    optimizer=optim.Adam,
    criterion=nn.CrossEntropyLoss
)

# Do not remove comments bellow if the model is already trained and saved

"""

net.fit(train_data, y=y_train)

val_loss=[]
train_loss=[]
for i in range(50):
    val_loss.append(net.history[i]['valid_loss'])
    train_loss.append(net.history[i]['train_loss'])
    
plt.figure(figsize=(10,8))
plt.semilogy(train_loss, label='Train loss')
plt.semilogy(val_loss, label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.show()    

# saving model
with open('saved_model_part1', 'wb') as f:
    pickle.dump(net, f)
"""
# loading model
with open('saved_model_part1', 'rb') as f:
    net = pickle.load(f)


y_pred = net.predict(test_dataset)
y_test = np.array([y for x, y in iter(test_dataset)]) 

print(f"Precision is: {100 * precision_score(y_test,y_pred, average='weighted'):.3f} %")
print(f"Accuracy is: {100 * accuracy_score(y_test,y_pred):.3f} %")
print(f"Recall is: {100 * recall_score(y_test,y_pred, average='weighted'):.3f} %")
print(f"F1 is: {100 * f1_score(y_test,y_pred, average='weighted'):.3f} %")

plot_confusion_matrix(net, test_dataset, y_test.reshape(-1, 1),display_labels= classes)
plt.show()



