#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from layer import Layer

class SoftmaxLoss(Layer):

    def __init__(self):
        pass

    def forward(self, X, y_true, train=None):
    	'''
    	Args:
    		X: input to this layer, a 2d tensor (batch_size, num_classes), each batch is probability distribution
            y_true: true labels

    	Return:
    		loss: cross entoropy loss
    	'''
        num_batch=X.shape[0]
        e = np.exp(X - np.max(X)) #numerical computation trick to avoid overflow 
        Y = e / np.array([np.sum(e, axis=1)]).T
        self.Y_cache=Y
        #citation: http://docs.chainer.org/en/stable/_modules/chainer/functions/loss/softmax_cross_entropy.html
        loss = -np.log(Y[xrange(len(y_true)), y_true]).sum(keepdims=True) / num_batch

    	return loss
        
    def backward(self, X, y_true):
    	'''
    	- juss pass the gradient from bottom layer to top layer
        - No parameter in this layer so just passing it!

    	Args:
    		X: input to this layer, a 2d tensor (batch_size, num_classes), each batch is probability distribution
            y_true: true labels
            #no propagated_grad because this is the last layer!!

    	Return:
    		propagating_grad: gradient that will be propagated to the bottom layer, a 2d tensor (batch_size, num_classes), same dimensional tensor as input
    	'''
        #softmax loss derivertive
        #citation
        #http://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function
        #http://docs.chainer.org/en/stable/_modules/chainer/functions/loss/softmax_cross_entropy.html
        num_batch=X.shape[0]
        propagating_grad = self.Y_cache.copy()
        propagating_grad[xrange(len(y_true)), y_true] -= 1
        
        return propagating_grad/num_batch