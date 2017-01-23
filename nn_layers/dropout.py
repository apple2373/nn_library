#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from layer import Layer

class Dropout(Layer):

    def __init__(self,drop_prob=0.5):
        self.drop_prob = drop_prob  #probability of keeping a unit active. higher = less dropout
        self.mask_cache=None

    def forward(self, X, train=True):
    	'''
    	Args:
    		X: input to this layer, any dimensional tensor
            train: if not training, it will output the 

    	Return:
    		Y: output to this layer, same dimensional tensor as input
    	'''
        #numpy syntax is inspired from http://docs.chainer.org/en/stable/_modules/chainer/functions/noise/dropout.html
        #implementation is inpired from http://cs231n.github.io/neural-networks-2/#reg

        if train:
            self.mask_cache=np.random.rand(*X.shape) < self.drop_prob
            Y=X*self.mask_cache
        else:
            Y=X*self.drop_prob

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
    	propagating_grad = propagated_grad * self.mask_cache

        return propagating_grad

    def update(self,learning_rate):
     	pass