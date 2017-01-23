#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from layer import Layer

class Sigmoid(Layer):

    def __init__(self):
        pass

    def forward(self, X, train=None):
    	'''
    	Args:
    		X: input to this layer, any dimensional tensor

    	Return:
    		Y: output to this layer, same dimensional tensor as input
    	'''
    	Y=1.0 / (1.0 + np.exp(-X))

    	return Y
        
    def backward(self, X, propagated_grad):
    	'''
    	- juss pass the dradient from bottom layer to top layer using ReLU
        - No parameter in this layer so just passing it!

    	Args:
    		X: input to this layer when forwarded last time, any dimensional tensor 
    		propagated_grad: gradient that is propagated from the top layer, same dimensional tensor as input

    	Return:
    		propagating_grad: gradient that will be propagated to the bottom layer, same dimensional tensor as input
    	'''
    	propagating_grad = propagated_grad * self.forward(X) * (1.0 - self.forward(X))

        return propagating_grad