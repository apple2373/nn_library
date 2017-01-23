#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from layer import Layer

class Linear(Layer):

    def __init__(self, in_dim, out_dim):
        #parameters
        self.W = np.random.normal(0, np.sqrt(1. / in_dim),(in_dim,out_dim)).astype(np.float32) # initialization scheme, citation: http://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network
        self.b = np.zeros(out_dim).astype(np.float32)#

        #gradients for parameters
        self.W_grad = np.zeros_like(self.W)
        self.b_grad = np.zeros_like(self.b)


    def forward(self, X, train=None):
    	'''
    	Args:
    		X: input to this linear transformation,  2d tensor (batch_size, in_dim)

    	Return:
    		Y: out put of lienar transformation, Y= XW + b,  2d tensor (batch_size, out_dim)
    	'''
    	Y=X.dot(self.W) + self.b

    	return Y
        
    def backward(self, X, propagated_grad):
    	'''
    	- Update gradient of parameters in this layer,  based on propagated_grad (gradient that is propagted from the top layer) 
    	- Compute the gradietn that will be prpagated to the bottom layer

    	Args:
    		X: input to this layer when forwarded last time,  2d tensor (batch_size, in_dim)
    		propagated_grad: gradient that is propagated from the top layer, 2d tensor (batch_size, out_dim)

    	Return:
    		propagating_grad: gradient that will be propagated to the bottom layer, 2d tensor (batch_size, in_dim)
    	'''
    	self.W_grad=X.T.dot(propagated_grad)
    	self.b_grad=propagated_grad.sum(0)  #=propagated_grad.dot(np.ones(out_dim,dtype=np.float32)) where out_dim=propagated_grad.shape[1]

    	# assert self.W_grad.shape==self.W.shape
    	# assert self.b_grad.shape==self.b.shape

    	propagating_grad = propagated_grad.dot(self.W.T)


        return propagating_grad

    def get_parms(self):
        #return parameters
        return [self.W,self.b]

    def get_grads(self):
        #return gradients with the same order as get_parms()!
        return [self.W_grad,self.b_grad]