import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
import torchvision


def get_mean_std_train_data(data_root):
    
    train_transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.CIFAR10(root=data_root, train=True, download=False, transform=train_transform)
    
    ### BEGIN SOLUTION
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=len(train_set),
        num_workers=1
    )
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in train_loader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    ### END SOLUTION
    
    return mean, std


def get_data(batch_size, data_root, num_workers=1):
    
    
    try:
        mean, std = get_mean_std_train_data(data_root)
        assert len(mean) == len(std) == 3
    except:
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        
    
    train_test_transforms = transforms.Compose([
        # this re-scale image tensor values between 0-1. image_tensor /= 255
        transforms.ToTensor(),
        # subtract mean and divide by variance.
        #transforms.Normalize(mean, std)
    ])
    
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
    return train_loader, test_loader, mean, std


#_________________________________________________

if torch.cuda.is_available():
	device = "cuda"
	print("CUDA !")
else:
	device = "cpu"
	num_workers_to_set = 2
	print("CPU !")


train_loader, test_loader, mean, std = get_data(batch_size=16, data_root="./")

print(len(train_loader))
#print(mean, std)

# Display image and label.
train_features, train_labels = next(iter(train_loader))
print(type(train_features), type(train_labels))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
#img = train_features[0].squeeze()
#label = train_labels[0]
img = torchvision.utils.make_grid(train_features)
img = img.numpy()
plt.imshow(np.transpose(img, (1, 2, 0)))
#plt.imshow(img[0], cmap="gray")
print(train_labels)
plt.show()

