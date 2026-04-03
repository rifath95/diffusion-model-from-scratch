import torch
import random

from torchvision import datasets, transforms

from config import *


# This transform just converts images to tensors
transform = transforms.ToTensor()

# Train dataset (will auto-download if not present)
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# Test dataset
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# Sampling a batch
def get_batch(split,digit=None):
    data = train_dataset if split == 'train' else test_dataset
    images2db = torch.zeros(batch_size,28,28)
    labelsb = torch.zeros(batch_size, dtype=torch.long)
    if digit is None:
        for i in range(batch_size):
            image , label = data[random.randint(0,len(data)-1)]
            images2db[i] = image.squeeze()
            labelsb[i] = label 
    else:
        for i in range(batch_size):
            while True:
                image , label = data[random.randint(0,len(data)-1)]
                if label == digit:
                    break
            images2db[i] = image.squeeze()
            labelsb[i] = label         
    images2db, labelsb = images2db.to(device), labelsb.to(device)
    return images2db, labelsb