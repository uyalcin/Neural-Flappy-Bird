import tensorflow as tf
import numpy as np
import random
from util import *
import random

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

# Create the placeholders

p_i = tf.placeholder(tf.int32)
p_j = tf.placeholder(tf.int32)
p_value = tf.placeholder(tf.float32)

#a_weights = tf.Variable([-1, N_neural], dtype=tf.float32)
a_weights = tf.Variable(0.0, validate_shape = False)
p_weights = tf.placeholder(a_weights.dtype)
assign = tf.assign(a_weights, p_weights, validate_shape = False)

assign_weights = tf.scatter_nd_update(a_weights, [[p_i, p_j]], p_value)

random_number = tf.random_normal([1], stddev=0.01)

def mutateNN(nn, sess):
    """ Take a neural network in parameters
    a change one of its weights """

    #Change weights 1
    #weights1 =
    sess.run(assign, {p_weights : nn.weights1})

    for i in range(N_features):
        for j in range(N_neural):
            if(random.randint(0, 0) == 0):
                init_value = 1 + weights1[i][j]
                init_value = 0
                # Make a small variation of the value
                mutate_value = init_value * sess.run(random_number)
                sess.run(assign_weights, {p_i : i, p_j : j, p_value : mutate_value})


    #print("assign i = " + str(i) + ", j = " + str(j))
    nn.weights1.assign(weights1)
    print(weights1)
    """
    #Change weights 2
    assign = tf.scatter_nd_update(nn.weights2, [[p_i, p_j]], p_value)

    for i in range(N_neural):
        for j in range(N_classes):
            if(random.randint(0, 0) == 0):
                init_value = 1 + nn.weights2[i][j]
                init_value = 0
                # Make a small variation of the value
                mutate_value = init_value * sess.run(random_number)
                sess.run(assign, {p_i : i, p_j : j, p_value : mutate_value})
                #print("assign i = " + str(i) + ", j = " + str(j))
    """
    #print(sess.run(nn.weights2))
    #print()


nn1 = NN()
nn2 = NN()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.graph.finalize()

    mutateNN(nn1, sess)

    print(sess.run(nn1.weights1))
    #print(sess.run(random_number))

