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


last_rnn_output = outputs[:, -1:]
final_output = get_linear_layer(last_rnn_output)

softmax = tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_output, labels=y)
cross_entropy = tf.reduce_mean(softmax)
train_step = tf.train.RMSPropOptimizer(0.001, 0.9) \
    .minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(final_output, 1))
