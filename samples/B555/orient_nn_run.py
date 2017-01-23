import math
import sys
import operator
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('../../../')

from nn_library.nn_layers.linear import Linear
from nn_library.nn_layers.relu import ReLU
from nn_library.nn_layers.sigmoid import Sigmoid
from nn_library.nn_layers.softmax import Softmax
from nn_library.nn_layers.softmax_loss import SoftmaxLoss
from nn_library.nn_layers.dropout import Dropout

from nn_library.nn_feedforward_network import FeedForwardNetwork

from nn_library.optimizers.sgd import SGD
from nn_library.optimizers.momentum_sgd import MomentumSGD
from nn_library.optimizers.adam import Adam


trainingFile = "train-data.txt"
testFile = "test-data.txt"

ORIENT_MAP={0:0,90:1,180:2,270:3}
ORIENT_MAP_id2orient={ v:k for (k,v) in ORIENT_MAP.iteritems()}

print "loading data..."

Xtrain = []
ytrain = []
with open(trainingFile, 'r') as f:
	lines = f.readlines()
	for line in lines:
		values = line.strip().split(' ')
		photo_id = values[0]
		orientation = ORIENT_MAP[int(values[1])]
		pixels = [int(p) for p in values[2:]]
		Xtrain.append(pixels)
		ytrain.append(orientation)

Xtest = []
ytest = []
with open(testFile, 'r') as f:
	lines = f.readlines()
	for line in lines:
		values = line.strip().split(' ')
		photo_id = values[0]
		orientation = ORIENT_MAP[int(values[1])]
		pixels = [int(p) for p in values[2:]]
		Xtest.append(pixels)
		ytest.append(orientation)

Xtrain=np.array(Xtrain,dtype=np.float32)/255
Xtest=np.array(Xtest,dtype=np.float32)/255
ytrain=np.array(ytrain)
ytest=np.array(ytest)

print "loading data... done"

print "training started!"

num_samples=Xtrain.shape[0]
input_dim=Xtrain.shape[1]
hidden_dim=2*input_dim
num_classes=4
batch_size=64
learning_rate=0.01
dropout_probability=0.5# probability of keeping a unit active. higher = less dropout

#define network
neural_net=FeedForwardNetwork()
#layer1
neural_net.add(Linear(input_dim,hidden_dim))
neural_net.add(Dropout(dropout_probability))
neural_net.add(ReLU())
#layer2
neural_net.add(Linear(hidden_dim,hidden_dim))
neural_net.add(Dropout(dropout_probability))
neural_net.add(ReLU())
#layer3
neural_net.add(Linear(hidden_dim,hidden_dim))
neural_net.add(Dropout(dropout_probability))
neural_net.add(ReLU())
#layer4
neural_net.add(Linear(hidden_dim,hidden_dim))
neural_net.add(Dropout(dropout_probability))
neural_net.add(ReLU())
#layer5
neural_net.add(Linear(hidden_dim,hidden_dim))
neural_net.add(Dropout(dropout_probability))
neural_net.add(ReLU())
#sotmax layer
neural_net.add(Linear(hidden_dim,num_classes))
neural_net.add(SoftmaxLoss())

#define optimizer 
optimizer=Adam()

#train
loss_list=[]
for epoch in range(10):
    print "epoch",epoch
    #suffle_data
    num_samples=Xtrain.shape[0]
    random_incides=np.random.permutation(num_samples)
    Xtrain,ytrain=Xtrain[random_incides],ytrain[random_incides]
    for i in range(0, num_samples/ batch_size):
        X_batch = Xtrain[i * batch_size: (i + 1) * batch_size]
        labels_batch= ytrain[i * batch_size: (i + 1) * batch_size]
        #forward
        loss=neural_net.forward(X_batch,labels_batch)
        # print loss in lively manner
        # if it is incresing or has NaN, maybe something is wrong 
        sys.stdout.write("\rsoftmax cross entorpy loss = %f"%loss)
        sys.stdout.flush()
        loss_list.append(loss)
        #backward
        neural_net.backward()
        #update
        optimizer.update_network_params(neural_net)

    sys.stdout.write("\r"+" "*50)
    sys.stdout.flush()
    sys.stdout.write("\r")
    sys.stdout.flush()

    #train accuracy
    Y=neural_net.forward(Xtrain,None,train=False)
    predictions=np.argmax(Y,axis=1)
    accuracy=1.0*sum(predictions==ytrain)/len(ytrain)
    print "train accuracy:",accuracy

    #test accuracy
    Y=neural_net.forward(Xtest,None,train=False)
    predictions=np.argmax(Y,axis=1)
    accuracy=1.0*sum(predictions==ytest)/len(ytest)
    print "test accuracy:",accuracy

#plot loss
plt.clf()
plt.plot(loss_list)
plt.savefig('nn_loss.png')

print "\ntraining done!"
print "let's check test accuracy!"

#check accuracy
#forward
Y=neural_net.forward(Xtest,None,train=False)
predictions=np.argmax(Y,axis=1)

#make confusion matrix
#citation:https://piazza.com/class/irnmu9v26th48r?cid=340
print "confusion matrix!"
confusion_mat = [ [ sum( [ (ytest[k], predictions[k]) == (col, row) for k in range(len(ytest)) ] ) for row in xrange(4) ] for col in xrange(4)  ] 
print "\n".join([ " ".join([ "%3d"%e for e in row]) for row in confusion_mat ] )
print "where index is: ",[ ORIENT_MAP_id2orient[idx] for idx in xrange(4)]

#compute accuracy
correct_count=sum([confusion_mat[i][i] for i in xrange(len(confusion_mat))])
accuracy=(1.0*correct_count/len(ytest))
print "accuracy is",accuracy