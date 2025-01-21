import numpy as np
import matplotlib as plt
import math


def generate_weights(prevLayerSize, currLayerSize, initEpsilon):
    initEpsilon = 1.0
    #prevents weights from being too low or too high
    return np.random.rand(currLayerSize, prevLayerSize) * (2 * initEpsilon) - initEpsilon

def bias(array):
    biasFreedom = 1.0
    return np.vstack([np.array([[biasFreedom]]), array])

def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))

def gradient(x):
    return sigmoid(x) * (1 - sigmoid(x))

def cost(pred, true):
    #cross entropy function (sum of the dot product)
    cost = np.sum(np.dot( - np.transpose(pred), np.log(true)) - np.dot(np.transpose(1-pred), np.log(1-true)))
    return cost


class neural_net:
    def __init__(self, layer_sizes):
        #layer_sizes = [input units, hidden units, output units]
        
        self.layerSizes = layer_sizes
        self.inputSize = layerSizes[0]
        self.outputSize = layerSizes[-1]
        self.numLayers = len(layerSizes)
        self.numHiddenLayers = len(layerSizes) - 2

        self.weights = []

    def forward_propogation(self, nx):
        bx = bias(x)
        layers = [bx, 0]
        for i in range (0,len(self.numLayers)-1):
            if i == len(self.numLayers -2):
                currLayer = sigmoid(np.dot(self.weights[i], layers[-1][0]))
                layers.append([currLayer, currLayer])
            else:
                if type(layers[-1]) == list:
                    zed = np.dot(self.weights[i], layers[-1][0])
                else:
                    zed = np.dot(self.weights[i], layers[-1])
                currLayer = addBiasTerm(sigmoid(zed))
                layers.append([currLayer, zed])
        return layers

    def back_prop(self, layers, val):
        evals = []
        gradients = []
        for i in range(0, len(self.weights)):
            shape = self.weights[i].shape
            ig = np.zeroes(shape)
            gradients.append(ig)
            
        for i in range(self.num_layers-1, 0, -1):
            if i == self.num_layers-1:
                
                throwEval = layers[-1][0] - val
                prevLayer = layers[i-1][0]
                
                n = np.dot(throwEval, np.transpose(prevLayer))
                gradientAccum[i-1] += n
                evals.append(throwEval)
            else:
                weight = self.weights[i]
                zed = layers[i][1]
                
                throwEval = np.dot(np.transpose(weight), evals[0])
                throwEval = np.multiply(throwEval, gradient(bias(zed)))
                throwEval = throwEval[1:, :]
                evals.insert(0, throwEval)

                n = np.dot(throwEval, np.transpose(layers[i-1][0]))
                gradients[i-1] += n
        return gradients

    def gradient_Descent(self, params, accumulation, rate):
        #Updates parameters after forward propogation
        for i in range(0, len(params)):
            params[i] = params[i] - (rate * accumulation[i])
        return params

    def train(self, inputMatrix, targetLabel, epochs, rate):
        '''
        For each epoch
            Forward Propagation -- Compute activations
            Cost Calculation -- Compute the cost for all data.
            Backpropagation -- Compute gradients
            Weight Update -- Update the weights using gradient descent
        The cost at each epoch is stored and returned

        '''
        costs = [] #outside loop to be returned

        for i in range(0, epochs):
            c = 0
            accumulation = []

            for j in range(0, len(self.weights)):
                accumulation.append(np.zeros(self.weights[j].shape))
                
            for j in range(0, numTrainingExamples):
                layerActivations = self.forward_prop(inputMatrix[j])
                
                c += cost_function(layerActivations[-1][0], targetLabel[j])
                
                bpgradients = self.back_prop(layerActivations, targetLabel[j])

                
                for k in range(0, len(accumulation)):
                    accumulation[k] += bpgradients[k]

                    
            for i in range(0, len(accumulation)):
                accumulation[i] /= len(inputMatrix)

                
            self.weights = self.update_params(self.weights, accumulation, rate)
            c = c * (1 / len(inputMatrix))
            costs.append(c)                 
        return costs

            
        
        