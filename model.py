import numpy as np
import random
from util import *
import random
from neural_network import NeuralNetwork

def randFactor():
    return 1 + (random.uniform(0, 1) - 0.5) * 3 + random.uniform(0, 1) - 0.5

def mutateNN(nn, mutate_rate):
    """ Take a neural network in parameters
    a change one of its weights """
    mutate_nn = NeuralNetwork(N_features, N_neurals, N_classes)
    #Change weights 1
    for i in range(N_features):
        for j in range(N_neurals):
            mutate_nn.weights1[i][j] = nn.weights1[i][j]
            if(random.randint(0, mutate_rate) == 0):
                init_value = 1 + (np.random.randn() / 10.0)
                # Make a small variation of the value
                mutate_nn.weights1[i][j] *= randFactor()

    #Change bias 1
    for i in range(N_neurals):
        mutate_nn.bias1[i] = nn.bias1[i]
        if(random.randint(0, mutate_rate) == 0):
            init_value = 1 + (np.random.randn() / 10.0)
            # Make a small variation of the value
            mutate_nn.bias1[i] *= randFactor()

    #Change weights 2
    for i in range(N_neurals):
        for j in range(N_classes):
            mutate_nn.weights2[i][j] = nn.weights2[i][j]
            if(random.randint(0, mutate_rate) == 0):
                init_value = 1 + (np.random.randn() / 10.0)
                # Make a small variation of the value
                mutate_nn.weights2[i][j] *= randFactor()

    #Change bias 2
    for i in range(N_classes):
        mutate_nn.bias2[i] = nn.bias2[i]
        if(random.randint(0, mutate_rate) == 0):
            init_value = 1 + (np.random.randn() / 10.0)
            # Make a small variation of the value
            mutate_nn.bias2[i] *= randFactor()

    return mutate_nn
