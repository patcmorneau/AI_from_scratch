import os
from typing import Iterable
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision
import matplotlib.pyplot as plt
import multiprocessing
import time


class MyModel(nn.Module):
	def __init__(self):
		super().__init__()
		
		self.conv1 = nn.Conv2d(3,6,5)
		self.conv2 = nn.Conv2d(6,16,5)
		self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)  # 6*6 from image dimension
		self.fc2 = torch.nn.Linear(120, 84)
		self.fc3 = torch.nn.Linear(84, 10)
	
	def forward(self, x):
		#print(x.shape)
		self.x1 = self.conv1(x)
		self.x2 = F.relu_(self.x1)
		self.x3 = F.max_pool2d(self.x2, 2)
		self.x4 = self.conv2(self.x3)
		self.x5 = F.relu_(self.x4)
		self.x6 = F.max_pool2d(self.x5, 2)
		
		self.x7 = self.x6.view(self.x6.size()[0], -1)
		self.x8 = self.fc1(self.x7)
		self.x9 = F.relu_(self.x8)
		self.x10 = self.fc2(self.x9)
		self.x11 = F.relu_(self.x10)
		self.x12 = self.fc3(self.x11)

		return self.x12
		

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


def train(device: str, model: nn.Module, optimizer: torch.optim.Optimizer, train_loader: torch.utils.data.DataLoader, epoch_idx: int):
	# change model in training mode
    model.train()
    
    # to get batch loss
    batch_loss = np.array([])
    
    # to get batch accuracy
    batch_acc = np.array([])
        
    for batch_idx, (data, target) in enumerate(train_loader):
        
        # clone target
        indx_target = target.clone()
        # send data to device (its is medatory if GPU has to be used)
        data = data.to(device)
        # send target to device
        target = target.to(device)

        # reset parameters gradient to zero
        optimizer.zero_grad()
        
        # forward pass to the model
        output = model(data)      
        
        # cross entropy loss
        loss = F.cross_entropy(output, target)
        
        # find gradients w.r.t training parameters
        loss.backward()
        # Update parameters using gardients
        optimizer.step()
        
        batch_loss = np.append(batch_loss, [loss.item()])
        
        # Score to probability using softmax
        prob = F.softmax(output, dim=1)
            
        # get the index of the max probability
        pred = prob.data.max(dim=1)[1]  
                        
        # correct prediction
        correct = pred.cpu().eq(indx_target).sum()
            
        # accuracy
        acc = float(correct) / float(len(data))
        
        batch_acc = np.append(batch_acc, [acc])

        if batch_idx % 10000 == 0 and batch_idx > 0:              
            print(
                'Train Epoch: {} [{}/{}] Loss: {:.6f} Acc: {:.4f}'.format(
                    epoch_idx, batch_idx * len(data), len(train_loader.dataset), loss.item(), acc
                )
            )
            
    epoch_loss = batch_loss.mean()
    epoch_acc = batch_acc.mean()
    return epoch_loss, epoch_acc


def validate(device, model: nn.Module, test_loader: torch.utils.data.DataLoader) -> float:
    
    model.eval()
    test_loss = 0
    count_corect_predictions = 0
    for data, target in test_loader:
        indx_target = target.clone()
        data = data.to(device)
        
        target = target.to(device)
        
        output = model(data)
        # add loss for each mini batch
        test_loss += F.cross_entropy(output, target).item()
        
        # Score to probability using softmax
        prob = F.softmax(output, dim=1)
        
        # get the index of the max probability
        pred = prob.data.max(dim=1)[1] 
        
        # add correct prediction count
        count_corect_predictions += pred.cpu().eq(indx_target).sum()

    # average over number of mini-batches
    test_loss = test_loss / len(test_loader)  
    
    # average over number of dataset
    accuracy = 100. * count_corect_predictions / len(test_loader.dataset)
    
    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, count_corect_predictions, len(test_loader.dataset), accuracy
        )
    )
    return test_loss, accuracy/100.0


############ MAIN ############
if torch.cuda.is_available():
	device = "cuda"
	print("CUDA !")
else:
	device = "cpu"
	num_workers_to_set = 2
	print("CPU !")


# init model
model = MyModel()
model.to(device)


# optimizer
optimizer = optim.SGD(model.parameters(),lr=0.05)
    

# get data
train_loader, test_loader = get_data(batch_size=1, data_root="./")


nbEpoch = 100

## Training ##

best_loss = torch.tensor(np.inf)
best_accuracy = torch.tensor(0)

# epoch train/test loss
epoch_train_loss = np.array([])
epoch_test_loss = np.array([])

# epch train/test accuracy
epoch_train_acc = np.array([])
epoch_test_acc = np.array([])

# trainig time measurement
t_begin = time.time()
for epoch in range(nbEpoch):
    
    train_loss, train_acc = train(device, model, optimizer, train_loader, epoch)
    
    epoch_train_loss = np.append(epoch_train_loss, [train_loss])
    
    epoch_train_acc = np.append(epoch_train_acc, [train_acc])

    elapsed_time = time.time() - t_begin
    speed_epoch = elapsed_time / (epoch + 1)
    speed_batch = speed_epoch / len(train_loader)
    eta = speed_epoch * nbEpoch - elapsed_time
    
    print(
        "Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(
            elapsed_time, speed_epoch, speed_batch, eta
        )
    )

    current_loss, current_accuracy = validate(device, model, test_loader)
    
    epoch_test_loss = np.append(epoch_test_loss, [current_loss])

    epoch_test_acc = np.append(epoch_test_acc, [current_accuracy])
    
    if current_loss < best_loss:
        best_loss = current_loss
    
    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        #print('Accuracy improved, saving the model.\n')
        #save_model(model, device)
        
            
print("Total time: {:.2f}, Best Loss: {:.3f}, Best Accuracy: {:.3f}".format(time.time() - t_begin, best_loss, best_accuracy))

#return model, epoch_train_loss, epoch_train_acc, epoch_test_loss, epoch_test_acc


