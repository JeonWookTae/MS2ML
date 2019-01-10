import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def check_dir(path=None):
    if not os.path.isdir(path):
        os.mkdir(path=path)


def get_data(data_dir):
    check_dir(path=data_dir)
    return input_data.read_data_sets(train_dir=data_dir, one_hot=True)


element_size = 28
time_steps = 28
num_classes = 10
batch_size = 128
hidden_size = 128

_inputs = tf.placeholder(tf.float32, shape=[None, time_steps, element_size])
y = tf.placeholder(tf.float32, shape=[None, num_classes])

rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
outputs, status = tf.nn.dynamic_rnn(rnn_cell, _inputs, dtype=tf.float32)

wl = tf.Variable(tf.truncated_normal([hidden_size, num_classes], mean=0, stddev=.01))
b1 = tf.Variable(tf.truncated_normal([num_classes], mean=0, stddev=.01))


def get_linear_layer(vector):
    return tf.matmul(vector, wl) + b1


last_rnn_output = outputs[:, -1, :]
final_output = get_linear_layer(last_rnn_output)

softmax = tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_output, labels=y)
cross_entropy = tf.reduce_mean(softmax)
train_step = tf.train.RMSPropOptimizer(0.001, 0.9) \
    .minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(final_output, 1))
accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32))) * 100

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

DATA_DIR = r'../data/mnist'
mnist = get_data(data_dir=DATA_DIR)

test_data = mnist.test.images[:batch_size].reshape((-1, time_steps, element_size))
test_label = mnist.test.labels[:batch_size]

for step in range(3001):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    batch_x = batch_x.reshape((batch_size, time_steps, element_size))
    sess.run(train_step, feed_dict={_inputs: batch_x, y: batch_y})

    if step % 1000 == 0:
        feed_dict = {_inputs: batch_x, y: batch_y}
        acc = sess.run(accuracy, feed_dict=feed_dict)
        loss = sess.run(cross_entropy, feed_dict=feed_dict)
        print("Iter {}, minibatch loss={:.6f}, training accuracy={:.6f}".format(
            step, acc, loss
        ))

print("Testing Accuracy: {}".format(
    sess.run(accuracy, feed_dict={_inputs: test_data, y: test_label})))