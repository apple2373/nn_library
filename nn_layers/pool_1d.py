#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from layer import Layer

class Pool1D(Layer):

    def __init__(self, pool_size,stride=0, pad="zero",pool_type="ave"):
        #parameters
        self.filter_size=pool_size
        self.stride=stride
        self.pad=pad
        if pool_type=="max":
            self.pool=np.max
        if pool_type=="ave":
            self.pool=np.mean
        

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

        Y=self.pool(temp,axis=2).T

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
        #X = np.hstack([X, np.zeros([batch_size,self.filter_size*2]).astype(np.float32)])
        propagated_grad = np.hstack([propagated_grad, np.zeros([batch_size,self.filter_size*2]).astype(np.float32)])

        # temp=np.zeros(shape=(vector_dim,batch_size,self.filter_size),dtype=np.float32)
        # for i in xrange(vector_dim):
        #     temp[i]=X[:,i:i+self.filter_size]

        temp_grad=np.zeros(shape=(vector_dim,batch_size,self.filter_size),dtype=np.float32)
        for i in xrange(vector_dim):
            temp_grad[i]=propagated_grad[:,i:i+self.filter_size]

    	propagating_grad = self.pool(temp_grad,axis=2).T

        return propagating_grad