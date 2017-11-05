import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def add_layer(layer_name, inputs, in_size, out_size, active_function=None):
    with tf.name_scope(layer_name):
        #Weight = tf.Variable(tf.random_uniform([in_size, out_size]))
        Weight = tf.Variable(tf.random_normal([in_size, out_size]))
        biases = tf.Variable(tf.zeros([1, out_size])) + 0.01
        tf.summary.histogram(layer_name + "/weights", Weight)
        tf.summary.histogram(layer_name + "/biases", biases)
        y = tf.matmul(inputs, Weight) + biases
        if active_function is None:
            outputs = y
        else:
            outputs = active_function(y)
        tf.summary.histogram(layer_name + "/outputs", outputs)
        return outputs

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope("inputs"):
    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])

layer1 = add_layer("layer1", xs, 1, 10, active_function=tf.nn.relu)

predict = add_layer("predict_layer", layer1, 10, 1, active_function=None)

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(predict - ys), reduction_indices=[1]))
    tf.summary.scalar('loss', loss)

#train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#train = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(loss)
train = tf.train.AdadeltaOptimizer(learning_rate=0.01, rho=0.9, epsilon=0.000001).minimize(loss)
#train = tf.train.AdamOptimizer(beta1=0.9, beta2=0.999, learning_rate=0.1, epsilon=0.000001).minimize(loss)
#train = tf.train.RMSPropOptimizer(learning_rate=0.01, decay=0.9).minimize(loss)

init = tf.global_variables_initializer()
#figure = plt.figure()
#ax = figure.add_subplot(1, 1, 1)
#ax.scatter(x_data, y_data)
#plt.ion()  #not block the code
#plt.show()
merged_summary = tf.summary.merge_all()


with tf.Session() as session:

    writer = tf.summary.FileWriter("logs/", session.graph)
    session.run(init)
    for step in range(5000):
        session.run(train, feed_dict={xs: x_data, ys: y_data})
        if step % 50 == 0:
            rs = session.run(merged_summary, feed_dict={xs: x_data, ys: y_data})
            print(session.run(loss, feed_dict={xs: x_data, ys: y_data}))
            writer.add_summary(rs, step)
            #index = np.random.randint(0, x_data.shape[0])
            #index = [0, 1, 2]
            #print(session.run(predict, feed_dict={xs: x_data[index]}))






