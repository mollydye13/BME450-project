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
        self.conv0 = nn.Conv2d(3, 32, 3)
        self.conv1 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 512, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 26 * 26, 128)
        self.fc2 = nn.Linear(128, 2)

    # goes through neural network
    def forward(self, x):
        #print("input:", x.shape)
        x = self.conv0(x) #[64, 32, 222, 222]
        x = self.conv1(x) #[64, 64, 220, 220]
        x = self.pool(F.relu(x)) #[64, 64, 110, 110]
        #print("after 1st pool:", x.shape)
        x = self.conv2(x) #[64, 64, 108, 108]
        #print("Check Shape after conv2:",x.shape)
        x = self.pool(F.relu(x)) #[64, 64, 54, 54]
        #print("after pool 2:", x.shape)
        x = self.conv3(x) #[64, 512, 52, 52]
        #print("after conv3:", x.shape)
        x = self.pool(F.relu(x)) #[64, 512, 26, 26]
        #print(x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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
epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1}\nstart at", time.time() - start_time,"\n-------------------------------")
    train_loop_result = []
    test_loop_result = []
    train_loop_result = train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop_result = test_loop(test_dataloader, model, loss_fn)

#save loss and accuracy results in an array
    loss_train.append(train_loop_result[0])
    accuracy_train.append(train_loop_result[1])
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


# Display first 15 images of failed moles


def print_fail_loop(dataloader, model):
    with torch.no_grad():
        incorrect_images=[]
        correct_class = []
        for X, y in dataloader:
            pred = model(X)
            _, predicted = torch.max(pred, 1)
            for i in range(len(predicted)):
                if predicted[i] != y[i]:
                    incorrect_images.append(X[i])
                    correct_class.append(y[i])
    return incorrect_images, correct_class
fail_img, correct_class = print_fail_loop(test_dataloader, model)

# Display first 30 images of failed moles
w=40
h=30
fig=plt.figure(figsize=(12, 8))
columns = 5
rows = 6

for i in range(1, columns*rows +1):
    ax = fig.add_subplot(rows, columns, i)
    image = np.transpose(fail_img[i].numpy(), (1, 2, 0))  # Transpose the dimensions to (height, width, channels)
    if correct_class[i] == 0:
        ax.title.set_text('True Benign, \n Classified Malignant')
    else:
        ax.title.set_text('True Malignant, \n Classified Benign')
    plt.imshow(image, interpolation='nearest')
plt.show()

