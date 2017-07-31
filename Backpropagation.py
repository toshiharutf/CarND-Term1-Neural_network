# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 01:53:50 2017

@author: Toshiharu
"""

import numpy as np


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    temp = sigmoid(x)
    return temp * (1 - temp)


x = np.array([0.5, 0.1, -0.2])
target = 0.6
learnrate = 0.5

w1 = np.array([[0.5, -0.6],
             [0.1, -0.2],
             [0.1, 0.7]])

w2 = np.array([0.1, -0.3])

## Forward pass
a1=x

a2=sigmoid(np.dot(a1, w1))

a3 = sigmoid(np.dot(a2, w2))

## Backwards pass
## TODO: Calculate error
error = (target-a3)

# TODO: Calculate error gradient for output layer
d3 = del_err_output = np.multiply(error,sigmoid_prime(np.dot(a2, w2)) )

# TODO: Calculate error gradient for hidden layer
d2 = del_err_hidden = np.multiply(d3*w2,sigmoid_prime(np.dot(a1, w1)) )

# TODO: Calculate change in weights for hidden layer to output layer
Delta_2 = delta_w_h_o = learnrate*d3*a2

# TODO: Calculate change in weights for input layer to hidden layer
Delta_1 = delta_w_i_h = a1[:,None]*learnrate*d2

print('Change in weights for hidden layer to output layer:')
print(delta_w_h_o)
print('Change in weights for input layer to hidden layer:')
print(delta_w_i_h)
