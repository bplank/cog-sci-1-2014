#!/usr/bin/env python
# simple perceptron classifier
import sys
import numpy as np

def train(X,y,iterations,alpha=1.0):
    # model with one weight for each feature plus additional bias term
    model = np.zeros(X.shape[1]+1)
    print "training..."
    # train model
    for iter in range(iterations):
        print "Iteration: ", (iter+1)
        for n in range(X.shape[0]) :
            # retrieve data point and add bias feature value
            d = np.append(X[n], [-1])
            # get dot product
            score = np.dot(model, d)
            # predict given current model
            pred = 1 if score > 0 else 0
            # correct weights if prediction is wrong
            if pred != y[n] :
                for j in range(len(model)) :
                    model[j] += alpha * (y[n] - pred) * d[j]
    print "done."
    return model

def predict(testInst,model):
    # predict label for test point
    # append bias and calculate activation
    d = np.append(testInst, [-1])
    score = np.dot(model, d)
    pred = 1 if score > 0 else 0
    return pred

def main():

    # toy example  0=spam, 1=non-spam
    labelNames = ["spam","non-spam"]
    # training data:
    trainingPoints = np.array([[1,0,0,0],[1,1,0,0],[1,0,1,0],[1,0,1,1],[0,0,0,1]])
    trainingLabels = np.array([0, 0, 0, 1, 1])
    # test data
    testPoint = np.array([0,1,0,1])

    # hyper-parameters
    alpha = 1.0
    iterations = 5

    # train model
    model = train(trainingPoints,trainingLabels,iterations,alpha)

    # predict label for test point
    pred = predict(testPoint,model)

    # print model and prediction
    print "model:", model
    print "predicted:", pred, labelNames[pred]


if __name__=="__main__":
    main()

