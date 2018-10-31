import tensorflow as tf 
import numpy as np
import random

# Parameters
N_features = 2
N_classes = 2

N_neural = 10
learning_rate = 10e-2

# Placeholders
x = tf.placeholder(tf.float32, [None, N_features], name='features')
y = tf.placeholder(tf.float32, [None, N_classes], name='labels')

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

nn = NN()
nn2 = NN()
a = nn.a

# Cost function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(a),reduction_indices=[1]))

# Optimizer
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

# Accuracy
correct_prediction = tf.equal(tf.argmax(a, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")

# Open training data
file = open("data", "r")
t_x = []
train_y = []

data0 = []
data1 = []
for line in file.readlines():
    dat = line.split("/")
    features = [float(dat[0]), float(dat[1])]
    classe = float(dat[2])
    if(classe == 0):
        data0 += [features]
    if(classe == 1):
        data1 += [features]

print(len(data0))
n_data_max = len(data1)
print(n_data_max)

for i in range(n_data_max):
    t_x += [data0[random.randint(0, len(data0) - 1)]]
    train_y += [[1, 0]]
    t_x += [data1[i]]
    train_y += [[0, 1]]

# Normalize data
train_x = []
max_features_1 = max(k[0] for k in t_x)
max_features_2 = max(k[1] for k in t_x)
for k in range(len(t_x)):
    train_x += [[t_x[k][0] / max_features_1, t_x[k][1] / max_features_2]]


#print(train_x[1])
#print(train_y[0])
test_x = [0.98, 0.74]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter("graph")
    writer.add_graph(sess.graph)
    merged_summary = tf.summary.merge_all()

    for i in range(10000):
        k = sess.run([train_step], feed_dict={x : train_x , y : train_y})

    """
    for i in range(10):
        j = i/ 10.0
        test_x = [j, j]
        print(test_x)
        pred = sess.run(tf.argmax(a, 1), feed_dict={x: [test_x]})
        print(pred)
    """
    #print(sess.run(nn.weights1))
    #print(tf.constant(tf.random_normal(shape=[])))
    #pred = sess.run(a, feed_dict={x: [test_x]})
    #print(pred)
