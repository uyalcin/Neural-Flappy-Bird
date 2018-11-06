import numpy as np
import random
from util import *
import random
from neural_network import NeuralNetwork

def mutateNN(nn):
    """ Take a neural network in parameters
    a change one of its weights """
    mutate_nn = NeuralNetwork(N_features, N_neurals, N_classes)
    #Change weights 1
    for i in range(N_features):
        for j in range(N_neurals):
            mutate_nn.weights1[i][j] = nn.weights1[i][j]
            if(random.randint(0, 5) == 0):
                init_value = 1 + (np.random.randn() + 10.0)
                # Make a small variation of the value
                mutate_nn.weights1[i][j] = np.random.randn()

    #Change bias 1
    for i in range(N_neurals):
        mutate_nn.bias1[i] = nn.bias1[i]
        if(random.randint(0, 5) == 0):
            init_value = 1 + (np.random.randn() / 10.0)
            # Make a small variation of the value
            mutate_nn.bias1[i] = np.random.randn()

    #Change weights 2
    for i in range(N_neurals):
        for j in range(N_classes):
            mutate_nn.weights2[i][j] = nn.weights2[i][j]
            if(random.randint(0, 5) == 0):
                init_value = 1 + (np.random.randn() / 10.0)
                # Make a small variation of the value
                mutate_nn.weights2[i][j] = np.random.randn()

    #Change bias 2
    for i in range(N_classes):
        mutate_nn.bias2[i] = nn.bias2[i]
        if(random.randint(0, 5) == 0):
            init_value = 1 + (np.random.randn() / 10.0)
            # Make a small variation of the value
            mutate_nn.bias2[i] = np.random.randn()

    return mutate_nn
