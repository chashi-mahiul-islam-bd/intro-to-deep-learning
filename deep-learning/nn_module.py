#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
use of nn module
----------------
Created on Mon Feb 15 21:14:37 2021

@author: chashi
"""
from torch import nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self): 
        super().__init__()
        
        # inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # output layer, 10 units - one for each digits
        self.output = nn.Linear(256, 10)
        
        # define sigmoid activation and softmax output
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # pass the input tensor trhough each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        
        return x



class Network2(nn.Module):
    def __init__(self): 
        super().__init__()
        
        # inputs to hidden layer linear transformation
        self.hidden = nn.Linear(784, 256)
        # output layer, 10 units - one for each digits
        self.output = nn.Linear(256, 10)

        
    def forward(self, x):
        # pass the input tensor trhough each of our operations
        x = F.sigmoid(self.hidden(x))
        x = F.softmax(self.output(x), dim=1)

        return x
    
model = Network()

model


