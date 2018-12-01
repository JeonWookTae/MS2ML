import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = '../data/mnist'
NUM_STEPS = 1000
MINIBATCH_SIZE = 100

data = input_data.read_data_sets(DATA_DIR, one_hot=True)


def get_placeholder(shape=None):
    return tf.placeholder(dtype=tf.float32, shape=shape)


def get_weight(shape=None):
    return tf.Variable(tf.zeros(shape=shape))


def get_predict(x, w):
    return tf.matmul(x, w)


def get_cross_entropy(pred, label):
    logit = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label)
    cross_entropy = tf.reduce_mean(logit)
    return cross_entropy

def tensor_graph():
    x = get_placeholder(shape=[None, 784])
    w = get_weight(shape=[784, 10])
    y_true = get_placeholder(shape=[None, 10])
