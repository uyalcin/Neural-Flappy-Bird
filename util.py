import tensorflow as tf

# Parameters
N_features = 2
N_classes = 2

N_neurals = 10
learning_rate = 10e-2

# Placeholders
x = tf.placeholder(tf.float32, [None, N_features], name='features')
y = tf.placeholder(tf.float32, [None, N_classes], name='labels')

N_weights = (N_features + N_classes) * N_neurals
