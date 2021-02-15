#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 16:39:12 2021

@author: chashi
"""

# y = f2( f1( x*W1 )W2 )

import torch 

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