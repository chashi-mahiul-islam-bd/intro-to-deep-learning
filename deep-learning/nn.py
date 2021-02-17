#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NN with PyTorch
---------------
Created on Mon Feb 15 17:08:14 2021

@author: chashi
"""
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


# define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=0.5, std=0.5),
                                ])

# download and load the training data 
trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)

# plotting one of the images
plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')


def activation(x):
    return 1/(1+torch.exp(-x))

# flatten the input images
inputs = images.view(images.shape[0], -1)

# parameters
# n_inputs = 64x784
# n_hidden = 784x256
# n_output = 256x10
w1 = torch.rand(784, 256)
b1= torch.rand(256)

w2 = torch.rand(256, 10)
b2 = torch.rand(10)

h = activation(torch.mm(inputs, w1) + b1)

out = torch.mm(h, w2) + b2

def softmax(x):
    # x.shape = 64x10
    # dim=1 means sum will be done on the rows
    # this line give a 1x64 shape tensor
    # view(-1,1) will make it 64x1 tensor
    # it will help us to divide every x of a sample by the sum of that row
    return torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1, 1)

probabilities = softmax(out)

# does it have the right shape? should be (64,10)
print(probabilities.shape)
# does it sum to1
print(probabilities.sum(dim=1))


