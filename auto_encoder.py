import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data", one_hot=False)

learning_rate = 0.1
batch_size = 256
train_epochs = 5


n_input = 784

n_hidden1 = 256
n_hidden2 = 128

weights = {
        "input_hidden1": tf.Variable(tf.random_normal([n_input, n_hidden1])),
        "input_hidden2": tf.Variable(tf.random_normal([n_hidden1, n_hidden2])),
        "output_hidden1": tf.Variable(tf.random_normal([n_hidden2, n_hidden1])),
        "output_hidden2": tf.Variable(tf.random_normal([n_hidden1, n_input]))
        }

biases = {
        "input_bias1": tf.Variable(tf.random_normal([n_hidden1])),
        "input_bias2": tf.Variable(tf.random_normal([n_hidden2])),
        "output_bias1": tf.Variable(tf.random_normal([n_hidden1])),
        "output_bias2": tf.Variable(tf.random_normal([n_input]))
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


def encoder(x, bn=False):
    if bn:
        with tf.variable_scope("encoder/layer1"):
            layer1_net = tf.matmul(x, weights['input_hidden1']) + biases['input_bias1']
            layer1_net_bn = batch_normalization(layer1_net, [0], n_hidden1)
            layer1 = tf.nn.sigmoid(layer1_net_bn)
        with tf.variable_scope("encoder/layer2"):
            layer2_net = tf.matmul(layer1, weights['input_hidden2']) + biases['input_bias2']
            layer2_net_bn = batch_normalization(layer2_net, [0], n_hidden2)
            layer2 = tf.nn.sigmoid(layer2_net_bn)
        return layer2
    else:
        layer1 = tf.nn.sigmoid(tf.matmul(x, weights['input_hidden1']) + biases['input_bias1'])
        layer2 = tf.nn.sigmoid(tf.matmul(layer1, weights['input_hidden2']) + biases['input_bias2'])
        return layer2

def decoder(x, bn=False):
    if bn:
        with tf.variable_scope("decoder/layer1"):
            layer1_net = tf.matmul(x, weights['output_hidden1']) + biases['output_bias1']
            layer2_net_bn = batch_normalization(layer1_net, [0], [n_hidden1])
            layer1 = tf.nn.sigmoid(layer2_net_bn)
        with tf.variable_scope("decoder/layer2"):
            layer2_net = tf.matmul(layer1, weights['output_hidden2']) + biases['output_bias2']
            layer2_net_bn = batch_normalization(layer2_net, [0], [n_input])
            layer2 = tf.nn.sigmoid(layer2_net_bn)

    else:
        layer1 = tf.nn.sigmoid(tf.matmul(x, weights['output_hidden1']) + biases['output_bias1'])
        layer2 = tf.nn.sigmoid(tf.matmul(layer1, weights['output_hidden2']) + biases['output_bias2'])
    return layer2


X = tf.placeholder(tf.float32, [None, n_input])

## if use batch_normalization, converge is fast
with tf.variable_scope("encoder"):
    en = encoder(X, bn=True)
with tf.variable_scope("decoder"):
    de = decoder(en, bn=True)

loss = tf.reduce_mean(tf.pow(X - de, 2))
tf.summary.scalar("loss", loss)
train = tf.train.AdamOptimizer(0.01).minimize(loss)
#train = tf.train.RMSPropOptimizer(0.01, decay=0.9).minimize(loss)
merged_summary = tf.summary.merge_all()
init = tf.global_variables_initializer()
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter("auto_encoder_logs/", sess.graph)
    sess.run(init)
    epoch = 0
    total_batch = int(mnist.train.num_examples/batch_size)
    while epoch < train_epochs:
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c, rs = sess.run([train, loss, merged_summary], feed_dict={X: batch_xs})
            if i % 5 == 0:
                print 'loss', c
                train_writer.add_summary(rs, i)
        epoch += 1

    #encode_decode_res = sess.run(de, feed_dict={X: mnist.test.images[: 10]})
    #f, a = plt.subplots(2, 10, figsize=(10, 2))
    #for i in range(10):
    #    a[0][i].imshow(np.reshape(mnist.test.images[i], [28, 28]))
    #    a[1][i].imshow(np.reshape(encode_decode_res[i], [28, 28]))
    #plt.show()



