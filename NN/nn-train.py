# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 17:24:01 2021

@author: Ragıp AKYÜREK
"""
import os
import csv 
import math
import time
import numpy as np

from layers import LinearLayer
from layers import ReLU
from layers import Sigmoid
from layers import SoftMax
from layers import CrossEntropyLoss

global ssflag
ssflag = False

    
class NeuralNetwork():
    
    def __init__(self, infeatures, outputclasses, loadweights = False):
        self.inputvectorlen = infeatures
        self.outputclasses = outputclasses 
    
        # Define network structure
        self.hiddenlayers = 2
        fc1_neurons = 2
        fc2_neurons = 2
        
        self.fc1 = LinearLayer(self.inputvectorlen, fc1_neurons)
        self.fc2 = LinearLayer(fc1_neurons, fc2_neurons)
        self.outlayer = LinearLayer(fc2_neurons, self.outputclasses)
        
        self.hiddenactivation = ReLU()
        self.outputactivation = SoftMax()
        
        # if loadweights....
        
    def forward(self, X):

        if ssflag:
            print(">> Before forward propagation")
            print("Fc1 weights and biases")
            print(self.fc1.weights)
            print(self.fc1.biases)
            
            print("Fc2 weights and biases")
            print(self.fc2.weights)
            print(self.fc2.biases)
            
            print("Outlayer weights and biases")
            print(self.outlayer.weights)
            print(self.outlayer.biases)

        # ---------------------------------
            
        X = self.hiddenactivation(self.fc1(X))
        if ssflag:
            print(">> forward fc1 out:", X)
        
        X = self.hiddenactivation(self.fc2(X))
        if ssflag:
            print(">> forward fc2 out:", X)
         
        X = self.outputactivation(self.outlayer(X))
            
         # --------------------------------
        if ssflag: 
             print(">> forward outlayer out (aftersoftmax):", X)
             print(">> FORWARD PROP IS DONE.")
        
        return X
    
    def backward(self, lossfun):
        if ssflag:
            print(">> CALCULATING BACKPROP...")
                    
        # calculates delta and gradient
        # returns delta for each weight
        delta = lossfun.backward() 
        if ssflag:
            print("Delta of last layer:",delta, "with shape", delta.shape)

        listofweights = [self.fc1.weights, self.fc2.weights, self.outlayer.weights]
        listoflinearout = [self.fc1.output, self.fc2.output]
        
        self._listofdelta = []
        self._listofdelta.append(delta)
        
        upper_delta = delta
        for l in reversed(range(self.hiddenlayers)):
            tmp_sum = np.matmul(upper_delta, listofweights[l+1])
            tmp_deriv = self.hiddenactivation.derivative(listoflinearout[l])

            upper_delta = np.multiply(tmp_sum, tmp_deriv) #elementwise multiplication
            if ssflag:
                print("Delta of layer",l," is:",upper_delta,  "with shape", upper_delta.shape) # ------------------- debug point --------
            
            self._listofdelta.append(upper_delta)
        
        if ssflag:
            print(">> After delta calc")
            print("Deltas")
            print(self._listofdelta)
        
        return #self._listofdelta
    
    def updateweights(self, xbatch, learningrate = 0.1):
            
        
        listoflinearout = [self.fc1.output, self.fc2.output]
        scalar = (1/xbatch.shape[0]) # to take the average
        
        # compute gradient
        listofgrad_b = []
        listofgrad_w = []
 
        for l in range(self.hiddenlayers + 1):
            if l == 0:
                grad_w = np.matmul(xbatch.T, self._listofdelta[self.hiddenlayers-l]) * scalar                       
            else:
                xi = self.outputactivation(listoflinearout[l-1])
                grad_w = np.matmul(xi.T, self._listofdelta[self.hiddenlayers-l]) * scalar
                    
            listofgrad_b.append(self._listofdelta[self.hiddenlayers-l].mean(axis = 0, keepdims=True))
            listofgrad_w.append(grad_w.T)
    
        listofweights = [self.fc1.weights, self.fc2.weights, self.outlayer.weights]
        listofbiases = [self.fc1.biases, self.fc2.biases, self.outlayer.biases]
        
        # update weights
        for l in range(self.hiddenlayers + 1):
            listofweights[l] += learningrate * listofgrad_w[l] # idk why but this version decreases loss while the other increases...
            listofbiases[l] += learningrate * listofgrad_b[l]
        
        if ssflag:
            print(">> After update weights")
            print("Fc1 weights and biases")
            print(self.fc1.weights)
            print(self.fc1.biases)
            
            print("Fc2 weights and biases")
            print(self.fc2.weights)
            print(self.fc2.biases)
            
            print("Outlayer weights and biases")
            print(self.outlayer.weights)
            print(self.outlayer.biases)
        
        return

      
''' -------------------------------- MAIN ---------------------------------- '''
if __name__ == "__main__":

    
    # --------------------- dummy -------------
    
    
    X = np.array([[0.5, 0.2],[0.1, 0.4]])
    #X = ( X - X.min() ) / ( X.max() - X.min() )
    y = np.array([0,1])
    
    features = 2 
    classes = 2
    
    my_NN = NeuralNetwork(features, classes)
    lossfun = CrossEntropyLoss()
    
    loss = 0
    for i in range(1):
        tmploss = loss
        prediction = my_NN.forward(X)
        loss = lossfun(prediction,y, classes)
        if i%10 ==0:
            print(">> Calculated loss is", loss)
        my_NN.backward(lossfun)
        if ssflag:
            print(">> Updating weights with 0.1 lr.")
        my_NN.updateweights(X, 0.0001)
        
        if abs(tmploss-loss) < 0.000001:
            print("Breaking at",i)
            print(tmploss,loss)
            break
    
    prediction = my_NN.forward(X)
    print(prediction)
    
    # ----------------- end dummy ------------------
