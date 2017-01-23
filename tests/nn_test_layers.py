#!/usr/bin/env python
# -*- coding: utf-8 -*-


#internal scrit to test layers
#no gradient test!! just shape.

import numpy as np
import sys
sys.path.append('../')#root of nn_library

from nn_layers.linear import Linear
from nn_layers.relu import ReLU
from nn_layers.sigmoid import Sigmoid
from nn_layers.softmax import Softmax
from nn_layers.softmax_loss import SoftmaxLoss
from nn_layers.dropout import Dropout
from nn_layers.conv_1d import Conv1D
from nn_layers.pool_1d import Pool1D


#test linear layer
#in_dim=5
#out_dim=4
#batch_size=2
l1=Linear(5,4)
X=np.random.rand(2,5)
Y=l1.forward(X)
propagated_grad=np.ones((2,4))
propagating_grad=l1.backward(X,propagated_grad)
assert propagating_grad.shape==X.shape

#test ReLU layer
relu=ReLU()
X=np.random.rand(2,5)
Y=relu.forward(X)
propagated_grad=np.random.rand(2,5)
propagating_grad=relu.backward(X,propagated_grad)
assert propagating_grad.shape==X.shape

#test sigmoid layer
sigmoid=Sigmoid()
X=np.random.rand(2,5)
Y=relu.forward(X)
propagated_grad=np.random.rand(2,5)
propagating_grad=sigmoid.backward(X,propagated_grad)
assert propagating_grad.shape==X.shape

#test linear + ReLU
#in_dim=5
#out_dim=4
#batch_size=2
l1=Linear(5,4)
relu=ReLU()
X=np.random.rand(2,5)
#forward
Y1=l1.forward(X)
Y2=relu.forward(Y1)
#backward
propagated_grad=np.ones((2,4))
propagating_grad=relu.backward(Y1,propagated_grad)
assert propagating_grad.shape==Y1.shape
propagated_grad=propagating_grad
propagating_grad=l1.backward(X,propagated_grad)
assert propagating_grad.shape==X.shape

#test linear + ReLU + liner + sigmoid
#in_dim=5
#out_dim=4
#batch_size=2
l1=Linear(5,4)
relu=ReLU()
l2=Linear(4,1)
sigmoid=Sigmoid()
#forward
X=np.random.rand(2,5)
Y1=l1.forward(X)
Y2=relu.forward(Y1)
Y3=l2.forward(Y2)
Y4=sigmoid.forward(Y3)
#backward
propagated_grad=np.ones((2,1))
propagating_grad=sigmoid.backward(Y3,propagated_grad)
assert propagating_grad.shape==Y3.shape
propagated_grad=propagating_grad
propagating_grad=l2.backward(Y2,propagated_grad)
assert propagating_grad.shape==Y2.shape
propagated_grad=propagating_grad
propagating_grad=relu.backward(Y1,propagated_grad)
assert propagating_grad.shape==Y1.shape
propagated_grad=propagating_grad
propagating_grad=l1.backward(X,propagated_grad)
assert propagating_grad.shape==X.shape

#test softmax layer
softmax=Softmax()
X=np.random.rand(2,5)
Y=softmax.forward(X)
propagated_grad=np.random.rand(2,5)
propagating_grad=softmax.backward(X,propagated_grad)
assert propagating_grad.shape==X.shape

#test softmax_loss layer
softmax_loss=SoftmaxLoss()
X=np.array([[0.1,0.8,0.1],[0.2,0.6,0.1]])#2 x 3
labels=np.array([1,0])
Y=softmax_loss.forward(X,labels)
assert Y.shape == (1,)
propagating_grad=softmax_loss.backward(X,labels)
assert propagating_grad.shape==X.shape


#test linear + ReLU + liner + softmax_loss
#in_dim=5
#out_dim=4
#l2 in_dim=4
#l2 out_dim=num_classes=3
#batch_size=2
#layers
l1=Linear(5,4)
relu=ReLU()
l2=Linear(4,3)
softmax_loss=SoftmaxLoss()
#data
X=np.random.rand(2,5)
labels=np.array([2,0])#either 0,1,or 2 
#forward
Y1=l1.forward(X)
Y2=relu.forward(Y1)
Y3=l2.forward(Y2)
Y4=softmax_loss.forward(Y3,labels)
assert Y4.shape == (1,)
#backward
propagating_grad=softmax_loss.backward(Y3,labels)
assert propagating_grad.shape==Y3.shape
propagated_grad=propagating_grad
propagating_grad=l2.backward(Y2,propagated_grad)
assert propagating_grad.shape==Y2.shape
propagated_grad=propagating_grad
propagating_grad=relu.backward(Y1,propagated_grad)
assert propagating_grad.shape==Y1.shape
propagated_grad=propagating_grad
propagating_grad=l1.backward(X,propagated_grad)
assert propagating_grad.shape==X.shape


#test dropout layer
dropout=Dropout(0.9)
X=np.random.rand(2,5)
Y=dropout.forward(X)
propagated_grad=np.random.rand(2,5)
propagating_grad=dropout.backward(X,propagated_grad)
assert propagating_grad.shape==X.shape

#test conv_1d
conv_1d=Conv1D(3)
X=np.random.rand(3,10)
Y=conv_1d.forward(X)
assert Y.shape==X.shape
propagated_grad=np.random.rand(3,10)
propagating_grad=conv_1d.backward(X,propagated_grad)

#test plling_1d
pool_1d=Pool1D(3)
X=np.random.rand(4,10)
Y=pool_1d.forward(X)
assert Y.shape==X.shape
propagated_grad=np.random.rand(4,10)
propagating_grad=pool_1d.backward(X,propagated_grad)

