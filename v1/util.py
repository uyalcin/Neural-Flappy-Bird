import tensorflow as tf

# Parameters
N_features = 2
N_classes = 2

N_neural = 10
learning_rate = 10e-2

# Placeholders
x = tf.placeholder(tf.float32, [None, N_features], name='features')
y = tf.placeholder(tf.float32, [None, N_classes], name='labels')

