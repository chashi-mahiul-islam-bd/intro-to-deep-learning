#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training a network
------------------
Created on Mon Feb 15 21:31:31 2021

@author: chashi
"""

# cost function = loss function

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=0.5, std=0.5),
                                ])

# download and load the training data 
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# build a feed-forward network
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128,  64),
                      nn.ReLU(),
                      nn.Linear(64, 10))

# define the loss
criterion = nn.CrossEntropyLoss()

# get our data 
images, labels = next(iter(trainloader))

# flatten images
images = images.view(images.shape[0], -1)

# forward pass, get out logits
logits = model(images)

# calculate the loss with the logits and the labels
loss  = criterion(logits, labels)

print(loss)
