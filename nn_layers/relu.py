#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from layer import Layer

class ReLU(Layer):

    def __init__(self):
        pass

    def forward(self, X, train=None):
    	'''
    	Args:
    		X: input to ReLU, any dimensional tensor

    	Return:
    		Y: output to ReLU, same dimensional tensor as input
    	'''
    	Y=np.maximum(0,X)

    	return Y
        
    def backward(self, X, propagated_grad):
    	'''
        - juss pass the gradient from bottom layer to top layer
        - No parameter in this layer so just passing it!

    	Args:
    		X: input to this layer when forwarded last time, any dimensional tensor 
    		propagated_grad: gradient that is propagated from the top layer, same dimensional tensor as input

    	Return:
    		propagating_grad: gradient that will be propagated to the bottom layer, same dimensional tensor as input
    	'''
    	propagating_grad = propagated_grad * (X > 0)

        return propagating_grad

    def update(self,learning_rate):
     	pass