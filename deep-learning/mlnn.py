#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi Layer Neural Network (MLNN)
----------------------------------
Created on Mon Feb 15 16:39:12 2021

@author: chashi
"""
# 3 X 2 X 1

# y = f2( f1( x*W1 )W2 )

import torch 
import slnn
### Generate some data
torch.manual_seed(7) # seet the random seed so things are predictable

# features are 3 random normal variables
features = torch.randn((1,3))

# define the size of each layer in our network
n_input = features.shape[1]     # number of input units, must match number of input features
n_hidden = 2                    # number of hidden units
n_output = 1                    # number of output units

# weights for inputs to hidden layer
W1 = torch.randn(n_input, n_hidden)
# weights for hidden layer to output layer
W2 = torch.randn(n_hidden, n_output)

# bias terms for hidden and output layers
B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))

h = slnn.activation(torch.mm(features, W1) + B1)
ouptut = slnn.activation(torch.mm(h, W2) + B2)


# Numpy to torch and back
import numpy as np

a = np.random.rand(4,3)
b = torch.from_numpy(a) #inpl
c = b.numpy()

# the memory is shared between the Numpyt and Torch tensor, so if you change 
# the values in-place of one object, the other will change as well
# multiply pytorch tensor by 2, in place
b.mul_(2)
a
