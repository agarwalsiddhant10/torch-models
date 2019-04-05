import torch
import torch.nn as nn
import torch.optim as optim
import numpy as numpy
import torchvision
from torchvision import models
from my_model import my_model
from dataset import dataset
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

print("Libraries loaded")

data_train = dataset('coco/images/', 'coco/annotations/person_keypoints_train2014.json')
train_loader =  DataLoader(data_train, batch_size = 8, shuffle = True)
data_val = dataset('coco/images/', 'coco/annotations/person_keypoints_val2014.json')
val_loader = DataLoader(data_val, batch_size = 8, shuffle = False)

print("Hopefully data bhi load ho gaya hoga")
# print(data_train)
num_classes = 51

model = models.resnet101(pretrained = True)

ct = 0
for child in model.children():
    ct += 1
    # print(ct)
    if(ct < 8):
        for param in child.parameters():
            param.requires_grad = False
    if(ct>=8):
        pass
        # print(child)
infeatures = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(infeatures, 1024), nn.Dropout(0.5))

model_ext = my_model()
model_final = nn.Sequential(model, model_ext)

print("Model bana gaya hoga")

print(model_final)
if torch.cuda.is_available():
    model_final = model_final.cuda()

optimizer = optim.Adam(model_final.parameters(), lr = 1e-4, betas = (0.9, 0.999), eps = 1e-8)

for epoch in range(200):
	epoch_train_loss = 0
	train_acc = 0
	print("Epoch Number {}".format(epoch))
	for i, data in enumerate(train_loader, 0):
		# print(i)
		imgs, keypoints = data
		# print("img: ", imgs.shape)
		imgs, keypoints = Variable(imgs), Variable(keypoints)
		imgs = imgs.float().cuda()
		keypoints = keypoints.float().cuda()

		model_final = model_final.train().cuda()

		outputs = model_final.forward(imgs).cuda()

		criterion = nn.MSELoss()
		loss = criterion(outputs, keypoints)
		print('Batch: {} Loss {}'.format(i, loss.item()))
		epoch_train_loss = epoch_train_loss + loss.item()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
    
	print('Epoch: %d\nLoss: %f' %(epoch +1, (1.0*epoch_train_loss)/len(train_loader) ))
    # print("Train accuracy: ", 100.0*train_acc)
    
    # correct_epoch_test = 0
  
    # for i_test, test_data in enumerate(val_load, 0):
    #     inputs_test, keypoints_test = test_data
    
    #     inputs_test, keypoints_test = Variable(inputs_test), Variable(keypoints_test)
    
    #     inputs_test = inputs_test.float().cuda()
    #     keypoints_test = keypoints_test.float().cuda()
    
    #     net = net.eval()
    
    #     outputs_test = net.forward(inputs_test).cuda()

	# 	error_test = torch.cumsum((outputs_test - keypoints_test), dim = 1).float().cuda()

    #     # correct_test = (error_test < ).float().sum()
    #     correct_epoch_test += correct_test.item()
    
    # correct_epoch_test /= (len(val_load)*8)
    # print("Test accuracy: ", 100.0*correct_epoch_test)

print('Finished Training')








