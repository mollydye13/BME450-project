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
#folder_train = 'C:/Users/zoeol/OneDrive/Documents/Spring 2024/BME 450 Deep Learning/Mole Cancer or Not_Project/Small_Data/train'
#folder_test = 'C:/Users/zoeol/OneDrive/Documents/Spring 2024/BME 450 Deep Learning/Mole Cancer or Not_Project/Small_Data/test'

# file destination Zoe Purdue Computer
# folder_train = 'C:/Users/zegbert/Mole_Project/BME450-project/train'
# folder_test = 'C:/Users/zegbert/Mole_Project/BME450-project/test'

# file destinations Molly
folder_train = 'C:/BME 450/train'
folder_test = 'C:/BME 450/test'

training_data = datasets.ImageFolder(
    root = folder_train,  # the folder where all the data is located 
    transform=ToTensor()
)
test_data = datasets.ImageFolder(
    root = folder_test,
    transform=ToTensor()
)

print("Done Defining Data Sets at", time.time() - start_time)

class MLP(nn.Module):
    '''
    Multilayer Perceptron.
    '''
# Define the Neural Net
# class Net(nn.Module):
    def __init__(self):
        #super(Net, self).__init__()
        super().__init__()
        print("begin flattening images at", time.time() - start_time)
        self.flatten = nn.Flatten() #makes the images into vectors (row by row)
        print("end flattening images at", time.time() - start_time)
        print("begin first linear layer at", time.time() - start_time)
        self.l1 = nn.Linear(3*224*224, 7500) #images are (224, 224, 3)
        self.bn1 = nn.BatchNorm1d(7500)     # batch normalization
#        self.l2 = nn.Linear(75000, 30000)
#        self.l3 = nn.Linear(75000, 30000)
#        print("begin 4th linear layer at", time.time() - start_time)
#        self.l4 = nn.Linear(30000, 15000)
#        self.l5 = nn.Linear(15000, 7500)
        self.l6 = nn.Linear(7500, 1000)
        self.bn2 = nn.BatchNorm1d(1000)     # batch normalization
        self.l7 = nn.Linear(1000, 512)
        self.l8 = nn.Linear(512,2)

    # goes through neural network
    def forward(self, x):
        x = self.flatten(x)
#        x = F.relu(self.l1(x))
        x = F.relu(self.bn1(self.l1(x)))
#        x = F.relu(self.l2(x))
#        print("Layer 2 Done")
#        x = F.relu(self.l3(x))
#        print("Layer 3 Done")
#        x = F.relu(self.l4(x))
#        print("Layer 4 Done")
#        x = F.relu(self.l5(x))
#        print("Layer 5 Done")
        x = F.relu(self.bn2(self.l6(x)))
#        x = F.relu(self.l6(x))
        x = F.relu(self.l7(x))
        output = self.l8(x)
        return output
    
    # implement L1 linearization
    def compute_L1_loss(self, w):
        return torch.abs(w).sum()

mlp = MLP()

#In the training loop specified subsequently, we specify a L1 weight, collect all parameters, compute L1 loss, and add it to the loss function before error backpropagation.
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    running_train_loss = 0.0
    total_train = 0
    correct_train = 0
    for batch, (X, y) in enumerate(dataloader):   # x is the data and y is teh target values
        # Compute prediction and loss
        optimizer.zero_grad()  # optimizer's gradients reset to zero
        pred = model(X)
        loss = loss_fn(pred, y) # calculates loss

        # compute L1 loss component
        L1_weight = 1.0
        L1_parameters = []
        for parameter in mlp.parameters():
            L1_parameters.append(parameter.view(-1))
        L1 = L1_weight * mlp.compute_L1_loss(torch.cat(L1_parameters))

        # add L1 loss component
        loss += L1

        # Backpropagation
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item() * X.size(0)
        
        _, predicted = torch.max(pred, 1)  # _, predicted = torch.max(pred.X, 1) didnt work
        total_train += y.size(0)
        correct_train += (predicted == y).sum().item() # comparing predicted to actual
    

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"training Accuracy: {(100 * correct_train / total_train):>0.1f}%, loss: {loss:>8f} L1 loss: {L1:>8f}\n")

            
    epoch_train_loss = running_train_loss / len(dataloader.dataset)   # abverage training loss per sample
    # train_loss_history.append(epoch_train_loss)
    train_acc = 100 * correct_train / total_train  
    # train_acc_history.append(train_acc)
    return [loss.item(), train_acc]


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    print(num_batches)
    print(size)
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    test_acc = correct*100
    return [test_loss, test_acc]

#Train the Neural Network
print("Model Initialization (Net) start at", time.time() - start_time)
# model = Net()
model = MLP()
print("Model Initialization (Net) done at", time.time() - start_time)

print("Data Loading start at", time.time() - start_time)
train_dataloader = DataLoader(training_data, shuffle = True, batch_size=64)  # batch size: number of training examples before model is updated
test_dataloader = DataLoader(test_data, shuffle = True, batch_size=64)
print("Data Loading done at", time.time() - start_time)

learning_rate = 1e-2
batch_size = 64

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

print("Epochs start at", time.time() - start_time)
epochs = 2
train_loss_history = []
test_loss_history = []
train_acc_history = []
test_acc_history = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop_result = train_loop(train_dataloader, model, loss_fn, optimizer)
    train_loss_history.append(train_loop_result[0])
    train_acc_history.append(train_loop_result[1])
    print("end train loop at", time.time() - start_time)
    test_loop_result = test_loop(test_dataloader, model, loss_fn)
    test_loss_history.append(test_loop_result[0])
    test_acc_history.append(test_loop_result[1])
print("done with both training loops at", time.time() - start_time)

# graph graphs displaying accuracy and loss over each epoch
print("!!!!!!!!!!!!!!!!!!!!!print!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
epochs_arr = range(1,epochs +1)
plt.figure(figsize=(10, 5))
plt.plot(epochs_arr, train_loss_history, label='Training Loss')  # fix this so that it appends
plt.plot(epochs_arr, test_loss_history, label='Testing Loss') # fix this so that it appends
plt.title('Training and Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(epochs_arr, train_acc_history, label='Training Accuracy')  # fix this so that it appends
plt.plot(epochs_arr, test_acc_history, label='Testing Accuracy') # fix this so that it appends
plt.title('Training and Testing Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
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


# Display first 15 images of failed moles
def print_fail_loop(dataloader, model):
    a = 0
    with torch.no_grad():
        incorrect_images=[]
        correct_class = []  # 0 if true benign, classified malignant; 1 if truely malignant but classified benign
        for X, y in dataloader:
            a += 1
            pred = model(X)
            _, predicted = torch.max(pred, 1)
            for i in range(len(predicted)):
                if predicted[i] != y[i]:
                    incorrect_images.append(X[i])
                    correct_class.append(y[i])
    return incorrect_images, correct_class, a


# Display first 30 images of failed moles
fail_img, correct_class, num_images = print_fail_loop(test_dataloader, model)

w=40
h=30
fig=plt.figure(figsize=(12, 8))
columns = 5
rows = 6
false_pos = 0
false_neg = 0

for i in range(1, columns*rows +1):
    ax = fig.add_subplot(rows, columns, i)
    image = np.transpose(fail_img[i].numpy(), (1, 2, 0))  # Transpose the dimensions to (height, width, channels)

    if correct_class[i] == 0:
        ax.title.set_text('True Benign, \n Classified Malignant')
        false_pos += 1
    else:
        ax.title.set_text('True Malignant, \n Classified Benign')
        false_neg += 1
    plt.imshow(image, interpolation='nearest')
plt.show()

# print stats about false positives and negatives
print('total number of images:    ', num_images)
print('incorrect classifications: ', len(correct_class), '%: ', (len(correct_class)/ num_images))
print('false positives:           ', false_pos, '%: ', (false_pos/ len(correct_class)))
print('false negatives:           ', false_neg, '%: ', (false_neg/ len(correct_class)))


ima = test_data[sample_num][0]
ima = (ima - ima.mean())/ ima.std()
iman = ima.permute(1, 2, 0) # needed to be able to plot
plt.imshow(iman)
