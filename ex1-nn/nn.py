#!/usr/bin/env python
## nearest neighbor classifier
import numpy as np
import sys

def train(X,y,distance):
    # what does it mean to train a NN model?
    # ...
    model=[X,y,distance]
    return model

def predict(testInst,model):
    # get info from model (later we will use a class rather than this way) 
    labels=model[1]
    instances=model[0]
    d=model[2]
    # compare ...
    distances=np.array([computeDistance(testInst,instance,d) for instance in instances])
    print >>sys.stderr, distances
    idx_argmin=np.argmin(distances)
    return labels[idx_argmin]

def computeDistance(inst1,inst2,d):
    if d=="euclidean":
        return np.sqrt(np.sum((inst1-inst2)**2))
    elif d=="manhattan":
        return np.sum(np.abs(inst1-inst2))
    else:
        raise Exception("invalid distance")

def main():

    # toy example  0=spam, 1=non-spam
    labelNames = ["spam","non-spam"]
    # training data:
    trainInstances = np.array([[0,1,0],[1,0,0],[1,0,1]])
    trainLabels = np.array([0, 1, 1])
    # test data:
    testInstance = np.array([0,1,1])

    # hyper-parameters
    #distance="euclidean"
    distance="manhattan"

    # train model
    model = train(trainInstances,trainLabels,distance)

    # predict label for test point
    pred = predict(testInstance,model)
    
    # print model and prediction
    print model
    print "predicted:", pred, labelNames[pred]

if __name__=="__main__":
    main()

    

