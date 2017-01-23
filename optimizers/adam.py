#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class Adam(object):

    def __init__(self, learning_rate=0.001, beta1=0.9,beta2=0.999,eps=1e-8):
        #parameters
        self.learning_rate=learning_rate
        self.beta1=beta1
        self.beta2=beta2
        self.eps=eps


    def change_param(learning_rate,momentum):
        #if you want to change parameters
        self.learning_rate=learning_rate
        self.momentum=momentum

    def update_layer_params(self,layer):
        params=layer["layer"].get_parms()#pointer to parameters
        grads=layer["layer"].get_grads()#pointer to gradients

        if "v" not in layer:
            layer["v"]=[]
            layer["m"]=[]
            for i in xrange(len(params)):
                layer["v"].append(0)
                layer["m"].append(0)

        for i in xrange(len(params)):
            #inspired from http://cs231n.github.io/neural-networks-3/
            layer["m"][i]=self.beta1*layer["m"][i] + (1-self.beta1)*grads[i]
            layer["v"][i]=self.beta2*layer["v"][i] + (1-self.beta2)*(grads[i]**2)
            params[i] += - self.learning_rate * layer["m"][i] / (np.sqrt(layer["v"][i]) + self.eps)

    def update_network_params(self,network):
        for i in xrange(len(network.layers)):
            if network.layers[i]["layer"].get_parms() is None:
                continue
            self.update_layer_params(network.layers[i])
