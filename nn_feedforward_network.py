#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class FeedForwardNetwork(object):

    '''
    This is a class to hold a simple feed forward network.
    No loop is allowed. Simply stack layer and the last layer must be loss layer.
    '''

    def __init__(self):
        #list to take care of layer connetction
        self.layers = []

    def add(self,layer):
        #add a layer on the top
        layer_info={}
        layer_info["layer"]=layer
        self.layers.append(layer_info)

    def forward(self,X,y_true,train=True):
        '''
        if train is true, it will give the loss value
        otherwise, the output will be the prdiction score
        '''
        for i in xrange(len(self.layers)-1):
            self.layers[i]["input"]=X
            X=self.layers[i]["layer"].forward(X,train)
        #last loss layer
        if train:
            self.layers[i+1]["input"]=X,y_true
            loss=self.layers[i+1]["layer"].forward(X,y_true,train)
            return loss
        else:
            return X

    def backward(self):
        num_layers=len(self.layers)
        Y,y_true=self.layers[num_layers-1]["input"]
        propagared_grad=self.layers[num_layers-1]["layer"].backward(Y,y_true)
        for i in xrange(2,len(self.layers)+1):
            Y=self.layers[num_layers-i]["input"]
            propagared_grad=self.layers[num_layers-i]["layer"].backward(Y,propagared_grad)

    def update(self,learning_rate):
        for i in xrange(len(self.layers)):
            self.layers[i]["layer"].update(learning_rate)

