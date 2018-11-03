import tensorflow as tf 
import numpy as np
import random
from util import *

class NN():
    """ 
    Simple implentation of neural network with Tensorflow 
    """
    def __init__(self):
        with tf.variable_scope("scope", reuse = tf.AUTO_REUSE):
            # Hidden layer
            self.weights1 = tf.Variable(tf.random_normal([N_features, N_neural]), name="weights1")
            self.bias1 = tf.Variable(tf.random_normal([N_neural]), name = "biases1")
        
            # ReLU activation
            self.hidden_layer = tf.nn.relu((tf.matmul(x, self.weights1) + self.bias1),name='hiddenLayer')
        
            # Output layer
            self.weights2 = tf.Variable(tf.random_normal([N_neural, N_classes]), name="weights2")
            self.bias2 = tf.Variable(tf.random_normal([N_classes]), name = "biases2")
            
            # Output
            self.a = tf.nn.softmax(tf.matmul(self.hidden_layer, self.weights2) + self.bias2, name='output')

