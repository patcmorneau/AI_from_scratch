import os
import time
from typing import Iterable
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision
#import matplotlib
#matplotlib.use("qt4agg")
import matplotlib.pyplot as plt
import multiprocessing

class MyModel(nn.Module):
	def __init__(self):
		super().__init__()
		
		#self.linear1 = nn.Linear(16 * 5 * 5, 120)
		#self.l8 = F.linear(torch.zeros([120]), torch.zeros([84, 120]))
		#self.l9 = F.linear(torch.zeros([84]), torch.zeros([10, 84]))
	
	def forward(self, x):
		print(x.shape)
		
		self.x1 = F.conv2d(x, torch.randn([6,3,5,5]))
		#print("x1", self.x1.shape)
		self.x2 = F.relu_(self.x1)
		#print("x2", self.x2.shape)
		#x3 = batchnorm
		self.x4 = F.max_pool2d(self.x2, 2)
		#print(self.x4.shape)
		self.x5 = F.conv2d(self.x4, torch.randn([16,6,5,5]))
		self.x6 = F.relu_(self.x5)
		self.x7 = F.max_pool2d(self.x6, 2)
		# flatten the output of conv layers
		# dimension should be batch_size * number_of weights_in_last conv_layer
		self.x8 = self.x7.view(self.x7.size()[0], -1)
		
		self.x9 = F.linear(torch.Tensor([16*5*5, 1]), torch.Tensor([120, 400]))
		#self.x9 = F.linear(torch.Tensor([self.x8.shape[1], 1]), torch.Tensor([120, self.x8.shape[1]]))
		self.x10 = F.relu_(self.x9)
		#self.x11 = F.linear(torch.Tensor([self.x10.shape[1], 1]), torch.Tensor([84, self.x10.shape[1]]))
		self.x11 = F.linear(torch.Tensor([120, 1]), torch.Tensor([84, 120]))
		self.x12 = F.relu_(self.x11)
		#self.x13 = F.linear(torch.Tensor([self.x12.shape[1], 1]), torch.Tensor([10, self.x12.shape[1]]))
		self.x13 = F.linear(torch.zeros([84]), torch.zeros([10, 84]))
		return self.x13


def get_data(batch_size, data_root, num_workers=1):
    # train dataloader
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=data_root, train=True, download=True, transform=transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    # test dataloader
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=data_root, train=False, download=True, transform=transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return train_loader, test_loader

def showWeights(rows):
	plt.figure(figsize=(20, 10))
	x=0
	for row in range(len(rows)):
		for column in range(len(rows[row])):
			x+=1
			if x == 7 or x == 23 or x == 39:
				x+=10
			plt.subplot(len(rows)+1, 16, x)
			mat = rows[row][column]
			mat = mat.numpy()
			plt.imshow(mat, cmap="gray")
	plt.show()

def showImage(img):
	img = img[0].squeeze()
	img = img.numpy()
	plt.imshow(np.transpose(img, (1, 2, 0)))
	plt.show()

############ MAIN ############
if torch.cuda.is_available():
	device = "cuda:0"
	print("CUDA !")
else:
	device = "cpu"
	num_workers_to_set = 2
	print("CPU !")


# init model
model = MyModel()
model.to(device)

# get data
train_loader, test_loader = get_data(batch_size=1, data_root="./")



train_features, train_labels = next(iter(train_loader))
#print(type(train_features), type(train_labels))
#print(f"Feature batch shape: {train_features.size()}")
#print(f"Labels batch shape: {train_labels.size()}")
job_for_another_core = multiprocessing.Process(target=showImage, args=(train_features,))
job_for_another_core.start()


# forward pass to the model
output = model(train_features)
#print(output.shape)
print(model.x1.shape)
"""
print(model.x2.shape)
print(model.x4.shape)
print(model.x5.shape)
print(model.x6.shape)
print(model.x7.shape)
"""



layers = [model.x1, model.x2, model.x4, model.x5, model.x6, model.x7]
columns = []
rows = []
for layer in layers:
	for channel in range(layer.shape[1]):
		columns.append(layer[0][channel])
	rows.append(columns)
	columns = []


job_for_another_core = multiprocessing.Process(target=showWeights, args=(rows,))
job_for_another_core.start()



