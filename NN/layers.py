# -*- coding: utf-8 -*-
"""
Created on Mar 10 2021

@author: bng
"""
import numpy as np

# ------------------------ Layer Object ---------------------------------------

global ssflag
ssflag = False

class LinearLayer:
    # A fully connected linear layer for MLP application
    def __init__(self, inputcount, neuroncount):
        self.neurons = neuroncount
        self.output = 0
         
        # initialize set of weights
        self.weights = np.random.random((neuroncount, inputcount)) * 0.1 # each row has weights of a single perceptron
        self.biases = np.random.random((1, neuroncount)) * 0.1

        self.weights = np.array([[1,1],[2,2]], dtype = float)
        self.biasestmp= np.array([0.1, 0.2])
      
        self.biases = np.reshape(self.biasestmp, self.biases.shape)
                                 
    def __call__(self, inputvector):
        self.output = np.dot(inputvector, self.weights.T) + self.biases 
        return self.output
    
# ------------------------ Hidden Layer Activations ---------------------------

# I defined each of the activation layers as classes as well, because it is faster to define derivative within themselves 
# instead of checking activation function names with if-else, though there could be better ways to implement this.
class ReLU:
    def __init__(self):
        # Computes ReLU activation for the output of each neuron (for a column vector)
        self.output = 0
    
    def __call__(self, inputvector):
        self.output = np.where(inputvector < 0, 0, inputvector)
        return self.output
        
    def derivative(self, inputvector):
        return np.where(inputvector <= 0, 0, 1)


class Sigmoid:
    def __init__(self):
        # Computes ReLU activation for the output of each neuron (for a column vector)
        self.output = 0
    
    def __call__(self, inputvector):
        self.output = 1 / (1 + np.exp(inputvector))
        return self.output
        
    def derivative(self, inputvector):
        sigmoid =  1 / (1 + np.exp(inputvector))
        return sigmoid * (1-sigmoid)


# ---------------------------- Output Activations -----------------------------
class SoftMax:
    def __init__(self):
        self.output = 0
    
    def __call__(self, inputvector): # input vector is 2D, each row corresponds to outlayer's output for the particular sample
    
        exps = np.exp(inputvector - inputvector.max(axis=1, keepdims=True))
        self.output = exps/np.sum(exps, axis=1, keepdims=True)
        
        # print(np.sum(exps, axis=1, keepdims=True),"*")
        return self.output
    
# -------------------------- Loss Functions ------------------------------
class CrossEntropyLoss:
    def __init__(self):
        self.loss = 0
        
    def __call__(self, probvector, labels, classcount):
        # Cross-entropy is a measure of the difference between two probability distributions for a given random variable or set of events (https://machinelearningmastery.com/cross-entropy-for-machine-learning/)
        # Be careful though, entropy take probabilistic values so it is compatible with softmax (or logit func) but it is not 
        # compatible with logsoftmax. If you are going to use LogSoftMax, you use NLLLoss (see https://stats.stackexchange.com/questions/436766/cross-entropy-with-log-softmax-activation).
        # https://deepnotes.io/softmax-crossentropy
        self.inputvector = probvector

        onehot = np.zeros((labels.size, classcount))
        for i in range(len(labels)):
            onehot[i,labels[i]] = 1
        
        self.onehot = onehot
        if ssflag:
            print("onehot",onehot)
        
        crossentropy = -np.log(np.sum(probvector * onehot, axis=1)) 
        
        if ssflag:
            print("crossent",crossentropy)
        # if (np.sum(probvector * onehot, axis=1)).all() == False:
        #     print(probvector,"---")
        
        self.loss = crossentropy.mean() 
        return self.loss
    
    def backward(self):
        self.dloss = self.inputvector - self.onehot # - 1 demiştim ama öle diilmiş ya
        
        # print(">>>> cokomelli")
        # print("prob vector shape", self.inputvector.shape, self.inputvector)
        # print("onehot shape", self.onehot.shape, self.onehot)
        # print()
        return self.dloss

    
    