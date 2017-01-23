#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class MomentumSGD(object):

    def __init__(self, learning_rate=0.01,momentum=0.9):
        #parameters
        self.learning_rate=learning_rate
        self.momentum=momentum
        self.velocity=0

    def change_param(learning_rate,momentum):
        #if you want to change parameters
        self.learning_rate=learning_rate
        self.momentum=momentum

    def update_layer_params(self,layer):
        params=layer["layer"].get_parms()#pointer to parameters
        grads=layer["layer"].get_grads()#pointer to gradients

        if "velocity" not in layer:
            layer["velocity"]=[]
            for i in xrange(len(params)):
                layer["velocity"].append(0)

        for i in xrange(len(params)):
            #inspired from http://cs231n.github.io/neural-networks-3/
            layer["velocity"][i] = self.momentum*layer["velocity"][i] - self.learning_rate*grads[i]
            params[i] += layer["velocity"][i]

    def update_network_params(self,network):
        for i in xrange(len(network.layers)):
            if network.layers[i]["layer"].get_parms() is None:
                continue
            self.update_layer_params(network.layers[i])
