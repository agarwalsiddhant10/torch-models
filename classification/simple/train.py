from dataset_class import *
from model import *
import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

data_train = dataset("X_train.npy", "y_train.npy")
data_val = dataset("X_val.npy", "y_val.npy")
train_load = DataLoader(data_train, batch_size = 8, shuffle = True)
val_load = DataLoader(data_val, batch_size = 8, shuffle = True)

net = Net()

if torch.cuda.is_available():
    net = net.cuda()

optimizer = optim.Adam(net.parameters(), lr= 3e-4, betas=(0.9,0.999), eps=1e-08)

for epoch in range(200):
    epoch_train_loss = 0
    train_acc = 0

    for i, data in enumerate(train_load, 0):
        imgs, labels = data

        imgs, labels = Variable(imgs), Variable(labels)
        imgs = imgs.float().cuda()
        labels = labels.float().cuda()

        net = net.train().cuda()

        outputs = net.forward(imgs).cuda()

        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, torch.max(labels,1)[1])
        outputs = torch.max(outputs, 1)[1]
        labels = torch.max(labels, 1)[1]

        correct = (outputs == labels).float().cuda()


        train_acc = train_acc + (correct.sum()).item()
        epoch_train_loss = epoch_train_loss + loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_acc /= (len(train_load)*8)
    print('Epoch: %d\nLoss: %f' %(epoch +1, (1.0*epoch_train_loss)/len(train_load) ))
    print("Train accuracy: ", 100.0*train_acc)
    
    correct_epoch_test = 0
  
    for i_test, test_data in enumerate(val_load, 0):
        inputs_test, labels_test = test_data
    
        inputs_test, labels_test = Variable(inputs_test), Variable(labels_test)
    
        inputs_test = inputs_test.float().cuda()
        labels_test = labels_test.float().cuda()
    
        net = net.eval()
    
        outputs_test = net.forward(inputs_test).cuda()
        outputs_test = torch.max(outputs_test, 1)[1]
        labels_test = torch.max(labels_test, 1)[1]

        correct_test = (outputs_test == labels_test).float().sum()
        correct_epoch_test += correct_test.item()
    
    correct_epoch_test /= (len(val_load)*8)
    print("Test accuracy: ", 100.0*correct_epoch_test)

print('Finished Training')
