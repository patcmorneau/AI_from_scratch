import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
import torchvision
import time


def get_data(batch_size, data_root, num_workers=1):

    train_test_transforms = transforms.Compose([transforms.ToTensor()])
    
    # train dataloader
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_test_transforms),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    # test dataloader
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=data_root, train=False, download=True, transform=train_test_transforms),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return train_loader, test_loader


#_________________________________________________

if torch.cuda.is_available():
	device = "cuda"
	print("CUDA !")
else:
	device = "cpu"
	num_workers_to_set = 2
	print("CPU !")


train_loader, test_loader = get_data(batch_size=16, data_root="./")
print(len(train_loader))
#show single image
"""
train_features, train_labels = next(iter(train_loader))
img = train_features[0].squeeze()
label = train_labels[0]
img = img.numpy()
plt.imshow(np.transpose(img, (1, 2, 0)))
print(train_labels)
plt.show()
"""

x = 0
plt.ion()
fig = plt.figure()
# Display images and labels.
for train_features, train_labels in train_loader:
	print(type(train_features), type(train_labels))
	print(f"Feature batch shape: {train_features.size()}")
	print(f"Labels batch shape: {train_labels.size()}")
	img = torchvision.utils.make_grid(train_features)
	img = img.numpy()
	plt.imshow(np.transpose(img, (1, 2, 0)))
	print(train_labels)
	fig.canvas.draw_idle()
	time.sleep(1.5)
	fig.canvas.flush_events()
	x += 1
	if x == 6:
		break;
