# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 18:10:19 2017

@author: Toshiharu
"""
import numpy as np

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    temp = sigmoid(x)
    return temp*(1-temp)

# Network size
N_input = 4
N_hidden = 3
N_output = 2

np.random.seed(42)
# Make some fake data
X = np.random.randn(4)

weights_input_to_hidden = np.random.normal(0, scale=0.1, size=(N_input, N_hidden))
weights_hidden_to_output = np.random.normal(0, scale=0.1, size=(N_hidden, N_output))
#print(weights_input_to_hidden.T)
#print(X[:,None])
#print(weights_hidden_to_output.T)

# TODO: Make a forward pass through the network

hidden_layer_in = np.dot(weights_input_to_hidden.T,X[:,None])
hidden_layer_out = sigmoid(hidden_layer_in)

print('Hidden-layer Output:')
print(hidden_layer_out)

output_layer_in = hidden_layer_out
output_layer_out = np.dot(weights_hidden_to_output.T,output_layer_in)

print('Output-layer Output:')
print(output_layer_out)