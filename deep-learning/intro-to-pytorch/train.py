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
import matplotlib.pyplot as plt
import numpy as np
# define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(0.5, 0.5),
                                ])

# download and load the training data 
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# with CrossEntropyLoss() -> it takes raw values 
# and then run LogSoftmax() and NLLLoss()
# build a feed-forward network
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128,  64),
                      nn.ReLU(),
                      nn.Linear(64, 10))

'''
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
'''

# with NLLLoss() -> it takes LogSoftmax values
# and then run NLLLoss()

# define the loss
criterion = nn.NLLLoss()

# get our data 
images, labels = next(iter(trainloader))

# flatten images
images = images.view(images.shape[0], -1)

# forward pass, get out logits
logps = model(images)

# calculate the loss with the logits and the labels
loss  = criterion(logps, labels)

print(loss)

print('Before backward pass: \n', model[0].weight.grad)

loss.backward()

print('After backward pass: \n', model[0].weight.grad)



# now we got our gradients. for updating the weights with the gradient we need:
# optimizers

# optimizers require the parameters to optimize and a leraning rate
from torch import optim
optimizer = optim.SGD(model.parameters(), lr = 0.01)


print ('Initial weights = ', model[0].weight)

images, labels = next(iter(trainloader))
images.view(64,784)


# clear the gradients, do this because gradients are accumulated
optimizer.zero_grad()
# define the loss
criterion = nn.NLLLoss()
# forward pass, then backward pass, then update weights
output = model.forward(images)
loss = criterion(output, labels)
loss.backward()
print('Gradient - ', model[0].weight.grad)

# take an update step and few the new weights
optimizer.step()
print('Updated weights - ', model[0].weight)
