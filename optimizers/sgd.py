#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class SGD(object):

    def __init__(self, learning_rate=0.01):
        #parameters
        self.learning_rate=learning_rate

    def change_param(learning_rate):
        #if you want to change parameters
        self.learning_rate=learning_rate

    def update_layer_params(self,layer):
        params=layer.get_parms()#pointer to parameters
        if params == None:
            return
        grads=layer.get_grads()#pointer to parameters
        for i in xrange(len(params)):
            params[i] -= self.learning_rate*grads[i]

    def update_network_params(self,network):
        for i in xrange(len(network.layers)):
            self.update_layer_params(network.layers[i]["layer"])
            