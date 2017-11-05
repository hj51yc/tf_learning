from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

def add_layer(layer_name, input_data, input_size, output_size, active_func=None):
    with tf.name_scope(layer_name):
        #W = tf.Variable(tf.random_uniform([input_size, output_size], minval=-1.0, maxval=1.0))
        W = tf.Variable(tf.random_normal([input_size, output_size], mean=0, stddev=0.5))
        bias = tf.Variable(tf.zeros([1, output_size])) + 0.01
        output_res = tf.matmul(input_data, W) + bias
        if active_func:
            return active_func(output_res)
        else:
            return output_res



mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

with tf.name_scope("inputs"):
    xs = tf.placeholder(tf.float32, [None, 784]) # 32 * 32

ys = tf.placeholder(tf.float32, [None, 10])

with tf.name_scope("hidden"):
    layer1 = add_layer("hidden", xs, 784, 32, tf.nn.tanh)
with tf.name_scope("predict"):
    predict = add_layer("predict", layer1, 32, 10, tf.nn.softmax)



### test ##########
with tf.name_scope("accuracy"):
    correct_predict = tf.equal(tf.argmax(predict, 1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

### train ############
with tf.name_scope("cross_entropy"):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(predict), reduction_indices=[1]))
    tf.summary.scalar('cross_entropy', cross_entropy)
#optimizer = tf.train.AdamOptimizer(beta1=0.9, beta2=0.999, learning_rate=0.01)
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()
session = tf.Session()

merged_summary = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("mnist_train_logs/", session.graph)
test_writer = tf.summary.FileWriter("mnist_test_logs/")
session.run(init)
for step in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    session.run(train, feed_dict={xs: batch_xs, ys: batch_ys})
    summary_rs = session.run(merged_summary, feed_dict={xs: batch_xs, ys: batch_ys})
    train_writer.add_summary(summary_rs, global_step=step) 
    if step % 50 == 0:

        rs, acc = session.run([merged_summary, accuracy], feed_dict={xs: mnist.test.images, ys: mnist.test.labels})
        print "accuracy", acc
        #train_writer.add_summary(rs)
        test_writer.add_summary(rs, global_step=step)

session.close()






