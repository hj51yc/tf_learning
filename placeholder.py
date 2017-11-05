import tensorflow as tf
import numpy as np

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1 : [1, 2], input2: [3, 4]}))


