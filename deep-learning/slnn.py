#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single Layer Neural Network (SLNN)
----------------------------------
Created on Mon Feb 15 16:05:20 2021

@author: chashi
"""

# y = f(w1x1 + w2x2 + b)

# tensors: a generalization of vectors or matrices

import torch 

def activation(x):
    """
    

    Parameters
    ----------
    x : torch.Tensor
        x is a tensor whose value is passed to get the activated value

    Returns
    -------
    returns the sigmoid activated value of x

    """
    return 1/(1+torch.exp(-x))


### Generate some data
torch.manual_seed(7) # seet the random seed so things are predictable

# features are 5 random normal variables
features = torch.randn((1,5))
# true weights for our data, random normal variables again
weights = torch.randn_like(features)
# and a true bias trem
bias = torch.rand((1,1))

# y = activation((features * weights).sum() + bias)
y = activation(torch.sum(features * weights) + bias)

# with matrix multiplication
# torch.mm() is better 
# as torch.matmul() will never throw error for wrong dimension
# wieghts need to be (5,1) instead of (1,5)

# weights.reshape(a,b) // returns a new tensor with the axB shape
# weights.resize_(a,b) // _ = inplace operation, 
# weights.view(a,b) // returns a new tensor, 
# view() returns an error if shape mismatch happens

weightsT = weights.view(5,1)

y = activation(torch.mm(features, weightsT) + bias)


