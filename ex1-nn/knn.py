#!/usr/bin/env python
## k-nearest neighbor sceleton
import numpy as np
import sys
from collections import Counter

def train(X,y,distance):
    # what does it mean to train a NN model?
    # ...
    model=[X,y,distance]
    return model

def predict(testInst,model,k):
    # get info from model (later we will use a class rather than this way) 
    labels=model[1]
    instances=model[0]
    d=model[2]
    # compare to every instance in training data
    distances=np.array([computeDistance(testInst,instance,d) for instance in instances])
    foundk=0
    print >>sys.stderr, distances
    neighbor_labels=[]

    ##### add code here #### 

    # as long as we haven't gotton k neighbors
        # find closest neighbor and save its label in the neighbor_labels list
        # remove index of closest point from distances and iterate

    ##### end #### 
    # the function returns the most frequent label in a very simple way (nb. ties are not broken randomly..) 
    return majority_vote(neighbor_labels)

def majority_vote(labels):
    return Counter(labels).most_common()[0][0] # get label of most frequent element

def computeDistance(inst1,inst2,d):
    if d=="euclidean":
        return np.sqrt(np.sum((inst1-inst2)**2))
    elif d=="manhattan":
        return np.sum(np.abs(inst1-inst2))
    else:
        raise Exception("invalid distance")

def main():

    # now we specify how many neighbors
    k = int(raw_input("Enter k: "))

    # toy example  0=square, 1=circle
    labelNames = ["square","circle"]
    # training data:

    #### add code here: read in the training instances from the file "knntrain.dat"

    trainInstances = ...
    trainLabels = ...

    #### end code

    # test data:
    testInstance = np.array([5.5,3.2])

    # hyper-parameters
    distance="euclidean"
    
    # train model
    model = train(trainInstances,trainLabels,distance)

    # predict label for test point
    pred = predict(testInstance,model,k)
    
    # print model and prediction
    print model
    print "predicted:", pred, labelNames[pred]

if __name__=="__main__":
    main()

    

