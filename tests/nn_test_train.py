 #!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
internal scrit to test neural net on a simple problem

test process goes as follows
- generate random 2d points with three classes
- 

'''
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')#root of nn_library

from nn_layers.linear import Linear
from nn_layers.relu import ReLU
from nn_layers.sigmoid import Sigmoid
from nn_layers.softmax import Softmax
from nn_layers.softmax_loss import SoftmaxLoss
from optimizers.sgd import SGD

def create_toy_data(N, d):
    x11 = np.random.randn(N / 2, d) / 5 + np.array([1, 1])
    x12 = np.random.randn(N / 2, d) / 5 + np.array([1.5, 0.5])
    x1 = np.vstack((x11, x12))
    x2 = np.random.randn(N, d) / 5 + np.array([2, 1])
    x3 = np.random.randn(N, d) / 5 + np.array([1.5, 1.5])
    plt.clf()
    plt.scatter(x1[:, 0], x1[:, 1], c='r')
    plt.scatter(x2[:, 0], x2[:, 1], c='g')
    plt.scatter(x3[:, 0], x3[:, 1], c='b')
    plt.savefig('nn_train_test_data.png')
    X = np.vstack((x1, x2, x3))

    d1 = np.ones((N,1))*0
    d2 = np.ones((N,1))*1
    d3 = np.ones((N,1))*2
    D = np.vstack((d1, d2, d3))

    # shuffle the order of sample points
    XD = np.hstack((X, D))
    np.random.shuffle(XD)
    X, D = XD[:, :d], XD[:, d:]

    return X, D.reshape(3*N).astype(np.int8)

N=1000
X, labels,=create_toy_data(N,2)

num_data=X.shape[0]
Xtrain=X[0:num_data-300]
Xtest=X[num_data-300:num_data]
ytrain=labels[0:num_data-300]
ytest=labels[num_data-300:num_data]

#test linear + ReLU + liner + softmax_loss
input_dim=2
hidden_dim=10
num_classes=3
batch_size=64
learning_rate=0.1
num_samples=Xtrain.shape[0]

#define layers
l1=Linear(input_dim,hidden_dim)
relu=ReLU()
l2=Linear(hidden_dim,num_classes)
softmax_loss=SoftmaxLoss()
softmax_activation=Softmax()

#optimizer
optimizer=SGD(learning_rate)


#train
loss_list=[]
for epoch in range(20):
    #suffle_data
    num_samples=Xtrain.shape[0]
    random_incides=np.random.permutation(num_samples)
    Xtrain,ytrain=Xtrain[random_incides],ytrain[random_incides]
    for i in range(0, num_samples/ batch_size):
        X_batch = Xtrain[i * batch_size: (i + 1) * batch_size]
        labels_batch= ytrain[i * batch_size: (i + 1) * batch_size]
        #forward
        Y1=l1.forward(X_batch)
        Y2=relu.forward(Y1)
        Y3=l2.forward(Y2)
        loss=softmax_loss.forward(Y3,labels_batch)
        print loss
        loss_list.append(loss)
        assert loss.shape == (1,)
        #backward
        propagated_grad=softmax_loss.backward(Y3,labels_batch)
        propagated_grad=l2.backward(Y2,propagated_grad)
        propagated_grad=relu.backward(Y1,propagated_grad)
        propagated_grad=l1.backward(X_batch,propagated_grad)
        #update
        optimizer.update_layer_params(l1)
        optimizer.update_layer_params(l2)

#plot loss
plt.clf()
plt.plot(loss_list)
plt.savefig('nn_train_test_loss.png')

#check accuracy
#forward
Y1=l1.forward(Xtest)
Y2=relu.forward(Y1)
Y3=l2.forward(Y2)
Y4=softmax_activation.forward(Y3)
predictions=np.argmax(Y4,axis=1)

truth_list=ytest
prediction_list=predictions

#make confusion matrix
#citation:https://piazza.com/class/irnmu9v26th48r?cid=340
CLASSES=[0,1,2]
confusion_mat = [ [ sum( [ (truth_list[k], prediction_list[k]) == (col, row) for k in range(len(truth_list)) ] ) for row in CLASSES ] for col in CLASSES  ] 
print "\n".join([ " ".join([ "%3d"%e for e in row]) for row in confusion_mat ] )
print "where index is: ",CLASSES

#compute accuracy
correct_count=sum([confusion_mat[i][i] for i in xrange(len(confusion_mat))])
accuracy=(1.0*correct_count/len(truth_list))
print "accuracy is",accuracy