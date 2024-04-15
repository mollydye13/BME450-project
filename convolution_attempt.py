from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

import os

#Import to Read and See Images
import matplotlib.pyplot as plt #need
import numpy as np
# import pandas as pd
# import os
# from glob import glob
# import seaborn as sns
# from PIL import Image

#Import to Make Neural Network
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import os

import time # I am using this to measure the time it takes to run certain things
start_time = time.time()
print("Done Importing Libraries")

# file destinations Zoe
#folder_train = 'C:/Users/zoeol/OneDrive/Documents/Spring 2024/BME 450 Deep Learning/Mole Cancer or Not_Project/data/train'
#folder_test = 'C:/Users/zoeol/OneDrive/Documents/Spring 2024/BME 450 Deep Learning/Mole Cancer or Not_Project/data/test'

# file destination Zoe Purdue Computer
folder_train = 'C:/Users/zegbert/Mole_Project/BME450-project/train'
folder_test = 'C:/Users/zegbert/Mole_Project/BME450-project/test'

# file destinations Molly
#folder_train = 'C:/BME 450/mole_cancerorno_reducedsize/train'
#folder_test = 'C:/BME 450/mole_cancerorno_reducedsize/test'

training_data = datasets.ImageFolder(
    root = folder_train,  # the folder where all the data is located 
    transform=ToTensor()
)
test_data = datasets.ImageFolder(
    root = folder_test,
    transform=ToTensor()
)

print("Done Defining Data Sets at", time.time() - start_time)

# Define the Neural Net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 48, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(48 * 24 * 24, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    # goes through neural network
    def forward(self, x):
        #print("input:", x.shape)
        x = self.conv1(x) #[64, 6, 220, 220]
        #print("after 1st conv:", x.shape)
        x = self.pool(F.relu(x)) #[64, 6, 110, 110]
        #print("after 1st pool:", x.shape)
        x = self.conv2(x) #[64, 16, 106, 106]
        #print("after conv2:",x.shape)
        x = self.pool(F.relu(x)) #[64, 16, 53, 53]
        #print("after pool 2:", x.shape)
        x = self.conv3(x)
        #print("after conv3:", x.shape)
        x = self.pool(F.relu(x))
        # print(x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #print(x.shape)
        return x


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_correct = 0

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute accuracy
        _, predicted = torch.max(pred, 1)
        num_correct += (predicted == y).sum().item()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    # Calculate accuracy
    accuracy = num_correct /size

    return [loss.item(), accuracy*100]

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return [test_loss, 100*correct]

#Train the Neural Network
print("Model Initialization (Net) start at", time.time() - start_time)
model = Net()
print("Model Initialization (Net) done at", time.time() - start_time)

print("Data Loading start at", time.time() - start_time)
train_dataloader = DataLoader(training_data, shuffle = True, batch_size=64)  # batch size: number of training examples before model is updated
test_dataloader = DataLoader(test_data, shuffle = True, batch_size=64)
print("Data Loading done at", time.time() - start_time)

learning_rate = 1e-2
batch_size = 64

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
accuracy_test = []
accuracy_train = []
loss_test = []
loss_train = []
print("Epochs start at", time.time() - start_time)
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop_result = []
    test_loop_result = []
    train_loop_result = train_loop(train_dataloader, model, loss_fn, optimizer)
    print("trainlopp result", train_loop_result)
    test_loop_result = test_loop(test_dataloader, model, loss_fn)

#save loss and accuracy results in an array
    loss_train.append(train_loop_result[0])
    accuracy_train.append(train_loop_result[1])
    print("end train loop at", time.time() - start_time)
    print("Train Accuracy:", accuracy_train)
    loss_test.append(test_loop_result[0])
    accuracy_test.append(test_loop_result[1])
    
print("done with both training loops at", time.time() - start_time)
# Plotting epoch vs loss
plt.figure()
plt.plot(range(1, t+2), loss_train, color = 'pink', label= "Training Loss")
plt.plot(range(1, t+2), loss_test, color = 'blue', label = "Testing Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch vs Loss')
plt.grid(True)
plt.legend()
plt.show()
# Plotting epoch vs accuracy
plt.figure()
plt.plot(range(1, t+2), accuracy_train, color = 'pink', label = "Training Accuracy")
plt.plot(range(1, t+2), accuracy_test, color = 'blue', label = "Testing Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Epoch vs Accuracy')
plt.grid(True)
plt.legend()
plt.show()
print("begin verification at", time.time() - start_time)
# Verify Training on One Image
categories = ["Benign", "Malignant"]
sample_num = 5 # select a specified sample in test_data

#Debugged with CHAT-GPT:
with torch.no_grad():
    inputs, labels = test_data[sample_num]
    inputs = inputs.unsqueeze(0)  # Add batch dimension
    r = model(inputs)
print("done debugging with chat gpt at", time.time() - start_time)

print('neural network output pseudo-probabilities:', r)
print('neural network output class number:', torch.argmax(r).item())
print('neural network output, predicted class:', categories[torch.argmax(r).item()])

# print(test_data[sample_num])
print('Inputs sample - image size:', test_data[sample_num][0].shape)
print('Label:', test_data[sample_num][1], '\n')


ima = test_data[sample_num][0]
ima = (ima - ima.mean())/ ima.std()
iman = ima.permute(1, 2, 0) # needed to be able to plot
plt.imshow(iman)
