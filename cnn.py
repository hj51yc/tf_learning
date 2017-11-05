import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, mean=0, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding="SAME")

xs = tf.placeholder(tf.float32, [None, 784])/255.0
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(xs, [-1, 28, 28, 1]) #[n_samples, 28, 28, channel]

with tf.name_scope("conv1"):
    W_conv1 = weight_variable([5, 5, 1, 32]) #[with, heigh, input_channel, output_channel]
    b_conv1 = bias_variable([32])
    conv_layer1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # 28 * 28 * 32
    print conv_layer1

with tf.name_scope("max_pool_2x2_layer1"):
    pool_layer1 = max_pool_2x2(conv_layer1) # 14 * 14 * 32
    print pool_layer1

with tf.name_scope("conv2"):
    W_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])
    conv_layer2 = tf.nn.relu(conv2d(pool_layer1, W_conv2) + b_conv2) # 14 * 14 * 64
    print conv_layer2

with tf.name_scope("max_pool_2x2_layer2"):
    pool_layer2 = max_pool_2x2(conv_layer2) # 7 * 7 * 64
    print pool_layer2

with tf.name_scope("full_nn"):
    W_func1 = weight_variable([7 * 7 * 64, 1024])
    b_func1 = bias_variable([1024])
    flat_func1 = tf.reshape(pool_layer2, [-1, 7 * 7 * 64])
    func1_layer = tf.nn.relu(tf.matmul(flat_func1, W_func1) + b_func1)
    drop_layer1 = tf.nn.dropout(func1_layer, keep_prob)
    print func1_layer

with tf.name_scope("predict_layer"):
    W_func2 = weight_variable([1024, 10])
    b_func2 = bias_variable([10])
    predict_layer = tf.nn.softmax(tf.matmul(drop_layer1, W_func2) + b_func2)
    #predict_layer = tf.nn.softmax(tf.matmul(func1_layer, W_func2) + b_func2)
    print predict_layer

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(predict_layer), reduction_indices=[1]))
tf.summary.scalar("loss", cross_entropy)

with tf.name_scope("accuracy"):
    predict_correct = tf.equal(tf.argmax(predict_layer, 1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(predict_correct, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

with tf.name_scope("train"):
    train = tf.train.AdamOptimizer(0.0001, beta1=0.9, beta2=0.999).minimize(cross_entropy)

saver = tf.train.Saver()
merged_summary = tf.summary.merge_all()
init = tf.global_variables_initializer()
with tf.Session() as session:
    train_writer = tf.summary.FileWriter("cnn_logs/", session.graph)
    saver.restore(session, "cnn_models/net1")
    session.run(init) ## when restore model, should comment this 
    for step in range(100):
        batch_x, batch_y = mnist.train.next_batch(50)
        loss, _ = session.run([cross_entropy, train], feed_dict={xs: batch_x, ys: batch_y, keep_prob:0.6})
        if step % 20 == 0:
            print "step", step, "loss", loss
            ts, acc = session.run([merged_summary, accuracy], feed_dict={xs: mnist.test.images, ys: mnist.test.labels, keep_prob:1.0})
            print 'accuracy', acc
            train_writer.add_summary(ts, global_step=step)
    path = saver.save(session, "cnn_models/net1")
    print "save to path ", path

            













