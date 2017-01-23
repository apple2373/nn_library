#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from layer import Layer

class Softmax(Layer):

    def __init__(self):
        pass

    def forward(self, X, train=None):
    	'''
    	Args:
    		X: input to this layer, any dimensional tensor

    	Return:
    		Y: output to this layer, same dimensional tensor as input
    	'''
        e = np.exp(X - np.max(X)) #numerical computation trick to avoid overflow 
        Y = e / np.array([np.sum(e, axis=1)]).T

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
        #softmax derivertive
        #citation
        #http://stackoverflow.com/questions/33541930/how-to-implement-the-softmax-derivative-independently-from-any-loss-function
    	#http://peterroelants.github.io/posts/neural_network_implementation_intermezzo02/s
        propagating_grad = X * propagated_grad
        s=propagating_grad.sum(axis=propagating_grad.ndim - 1, keepdims=True)
        propagating_grad -= X*s
        
        return propagating_grad