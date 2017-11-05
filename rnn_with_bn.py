import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

learning_rate = 0.001
batch_size = 128

n_input = 28
n_step = 28

n_hidden_unit = 128
n_classes = 10

x = tf.placeholder(tf.float32, [None, n_step, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

weights = {
        "in": tf.Variable(tf.random_normal([n_input, n_hidden_unit])),
        "out": tf.Variable(tf.random_normal([n_hidden_unit, n_classes]))
        }

biases = {
        "in": tf.Variable(tf.constant(0.1, shape=[n_hidden_unit])),
        "out": tf.Variable(tf.constant(0.1, shape=[n_classes]))
        }

def batch_normalization(x, axes, scale_shape):
    x_mean, x_dev = tf.nn.moments(x, axes)
    scale = tf.get_variable("scale", scale_shape, initializer=tf.constant_initializer(1.0), dtype=tf.float32)
    shift = tf.get_variable("shift", scale_shape, initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    epsilon = 0.0001
    
    ema = tf.train.ExponentialMovingAverage(decay=0.5)
    def mean_var_update():
        ema_apply_op = ema.apply([x_mean, x_dev])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(x_mean), tf.identity(x_dev)
    mean, var = mean_var_update()
    return tf.nn.batch_normalization(x, mean, var, scale, shift, epsilon)


def RNN(X, weights, biases):
    ## change [batch_size, n_step, n_input] to [batch_size * n_step, n_input]
    X = tf.reshape(X, [-1, n_input])
    #X_bn = batch_normalization(X, [0], [n_input])
    #X_in = tf.matmul(X_bn, weights['in']) + biases['in']
    X_in = tf.matmul(X, weights['in']) + biases['in']
    X_in = tf.reshape(X_in, [-1, n_step, n_hidden_unit]) #to shape [batch_size, n_step, n_hidden_unit]


    #LSTM
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_unit, forget_bias=1.0, state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)

    ##output layer
    outputs = tf.unstack(tf.transpose(outputs, perm=[1, 0, 2])) # shape list([batch_size, n_hidden_unit])
    result = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return result


pred = RNN(x, weights, biases)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for step in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_step, n_input])
        sess.run([train,], feed_dict={x: batch_xs, y: batch_ys})
        if step % 100 == 0:
            print(sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys}))






