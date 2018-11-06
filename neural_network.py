"""
Simple implementation of a neural network with one hidden layer
"""

import numpy as np

class NeuralNetwork():
    def __init__(self, _n_features, _n_neural, _n_classes):
        self.n_features = _n_features
        self.n_neurals = _n_neural
        self.n_classes = _n_classes

        # Hidden layer
        self.weights1 = np.random.randn(self.n_features, self.n_neurals)
        self.bias1 = np.random.randn(self.n_neurals)

        # Output layer
        self.weights2 = np.random.randn(self.n_neurals, self.n_classes)
        self.bias2 = np.random.randn(self.n_classes)

    def activate(self, x):
        self.hidden_layer = np.dot(x, self.weights1) + self.bias1
        self.activation_hidden_layer = self.relu(self.hidden_layer)

        self.output_layer = np.dot(self.activation_hidden_layer, self.weights2) + self.bias2
        self.activation_output_layer = self.softmax(self.output_layer)

        return self.activation_output_layer

    def predict(self, x):
        activation = self.activate(x)
        return np.argmax(activation, 0)

    def relu(self, value):
        return np.maximum(value, 0)

    def softmax(self, value):
        s = np.exp(value - np.max(value))
        return s / s.sum(axis=0)

""" 
Test
"""
nn = NeuralNetwork(2, 10, 2)

i = [0.0, 0.0]
print(nn.activate(i))
print(nn.predict(i))
