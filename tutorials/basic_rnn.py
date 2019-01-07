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
