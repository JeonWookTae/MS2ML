import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def check_dir(path=None):
    if not os.path.isdir(path):
        os.mkdir(path=path)


def get_data(data_dir):
    check_dir(path=data_dir)
    return input_data.read_data_sets(train_dir=data_dir, one_hot=True)


LOG_DIR = r"../data/logs/RNN_with_summaries"
DATA_DIR = r'../data/mnist'

time_steps = 28
element_size = 28
num_classes = 10
batch_size = 128
hidden_size = 128
data = get_data(data_dir=DATA_DIR)
_inputs = tf.placeholder(tf.float32, shape=[None, time_steps, element_size], name='inputs')
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='labels')

batch_x, batch_y = data.train.next_batch(batch_size)
batch_x = batch_x.reshape((batch_size, time_steps, element_size))
