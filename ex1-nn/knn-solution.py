import numpy as np
import sys
from collections import Counter

def train(X,y,distance):
    # what does it mean to train a NN model?
    # ... ==> no training, just memorizing instances (lazy learning)
    model=[X,y,distance]
    return model

def predict(testInst,model,k):
    # get info from model (later we will use a class rather than this way) 
    labels=model[1]
    instances=model[0]
    d=model[2]
    # compare to every instance in training data and store list of distances
    distances=np.array([computeDistance(testInst,instance,d) for instance in instances])
    foundk=0
    # in the neighbor_labels list we want to store the labels of the k-nearest neighbors
    neighbor_labels=[]
    # as long as we haven't found k nearest neighbors, do:
    while foundk < k:
        # find nearest one (minimum distance)
        idx_argmin=np.argmin(distances)
        # store label of closest one in list
        neighbor_labels.append(labels[idx_argmin])
        foundk+=1
        # remove item from list (or: keep only those which are not the closest one) 
        distances = [x for idx,x in enumerate(distances) if idx!=idx_argmin]
        # alternative as suggested in class: rather than removing, set distance to a high value
    # now we pass the k labels to the majority vote function that returns the most frequent label
    return majority_vote(neighbor_labels)

def majority_vote(labels):
    return Counter(labels).most_common()[0][0] # get label of most frequent element (note: ties are not broken arbitrarily here yet...)

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

    trainInstances = np.array([[ 5.1,  3.5],[ 4.9,  3. ],[ 4.7,  3.2],[ 4.6,  3.1],[ 5. ,  3.4],[ 6.7,  3. ],[ 6.3,  2.5],[ 6.5,  3. ],[ 6.2,  3.4],[ 5.9,  3. ]])
    trainLabels = np.array([0, 0,0,0,0,1,1,1,1,1])
    # test data:
    testInstance = np.array([5.5,3.2])

    # hyper-parameters
    distance="euclidean"
    
    # train model
    model = train(trainInstances,trainLabels,distance)

    # predict label for test point
    pred = predict(testInstance,model,k)
    
    # print model and prediction
    #print model
    print "predicted:", pred, labelNames[pred]

if __name__=="__main__":
    main()

    

