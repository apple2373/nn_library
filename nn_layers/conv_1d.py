#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from layer import Layer

class Conv1D(Layer):

    def __init__(self, filter_size,stride=0, pad="zero"):
        #parameters
        self.w = np.random.normal(0, np.sqrt(2.0 / filter_size),(filter_size)).astype(np.float32) # initialization scheme, citation: http://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network
        self.b = np.zeros(filter_size).astype(np.float32)#bias initialized to zero

        self.filter_size=filter_size
        self.stride=stride
        self.pad=pad

        #gradients for parameters
        self.w_grad = np.zeros_like(self.w)
        self.b_grad = np.zeros_like(self.b)


    def forward(self, X, train=None):
    	'''
    	Args:
    		X: input to this linear transformation,  2d tensor (batch_size, any_dim)

    	Return:
    		Y: out put of lienar transformation, Y= XW + b,  2d tensor (batch_size, out_dim)
    	'''
        batch_size,vector_dim = X.shape
        X = np.hstack([X, np.zeros([batch_size,self.filter_size*2]).astype(np.float32)])
        temp=np.zeros(shape=(vector_dim,batch_size,self.filter_size),dtype=np.float32)
        for i in xrange(vector_dim):
            temp[i]=X[:,i:i+self.filter_size] 

        Y= temp.dot(self.w).T
        
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
        batch_size,vector_dim = X.shape
        X = np.hstack([X, np.zeros([batch_size,self.filter_size*2]).astype(np.float32)])
        propagated_grad = np.hstack([propagated_grad, np.zeros([batch_size,self.filter_size*2]).astype(np.float32)])

        temp=np.zeros(shape=(vector_dim,batch_size,self.filter_size),dtype=np.float32)
        for i in xrange(vector_dim):
            temp[i]=X[:,i:i+self.filter_size]

        temp_grad=np.zeros(shape=(vector_dim,batch_size,self.filter_size),dtype=np.float32)
        for i in xrange(vector_dim):
            temp_grad[i]=propagated_grad[:,i:i+self.filter_size] 

    	self.w_grad=(temp*temp_grad).sum(0).sum(0)/vector_dim
        self.b_grad=temp_grad.sum(0).sum(0)/vector_dim

    	assert self.w_grad.shape==self.w.shape
        assert self.b_grad.shape==self.b.shape

    	propagating_grad = temp_grad.dot(self.w).T

        return propagating_grad

    def get_parms(self):
        #return parameters
        return [self.w,self.b]

    def get_grads(self):
        #return gradients with the same order as get_parms()!
        return [self.w_grad,self.b_grad]