import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os


def check_dir(path=None):
    if not os.path.isdir(path):
        os.mkdir(path=path)


def get_data(data_dir):
    check_dir(path=data_dir)
    return input_data.read_data_sets(train_dir=data_dir, one_hot=True)


def placeholder(shape=None):
    return tf.placeholder(dtype=tf.float32, shape=shape)


def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv_layer(input, shape):
    w = weight_variable(shape=shape)
    b = bias_variable(shape=[shape[3]])
    return tf.nn.relu(conv2d(input, w) + b)


def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    w = weight_variable([in_size, size])
    b = bias_variable([size])
    return tf.matmul(input, w) + b


def optimizer(logit, label, lr):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=label))
    adam_optimizer = tf.train.AdamOptimizer(lr)
    return adam_optimizer.minimize(cross_entropy)


def test_accuracy(pred, label):
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(label, 1))
    accuray = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuray


def main():
    IMAGE_SIZE = 784
    LABEL_SIZE = 10
    LEARNING_LATE = 0.001
    NUM_STEPS = 1000
    MINIBATCH_SIZE = 50
    DATA_DIR = r'../data/mnist'

    data = get_data(data_dir=DATA_DIR)
    x = placeholder(shape=[None, IMAGE_SIZE])
    y = placeholder(shape=[None, LABEL_SIZE])

    x_image = tf.reshape(x, [-1, 28, 28, 1])
    conv1 = conv_layer(x_image, shape=[5, 5, 1, 32])
    conv1_pool = max_pool_2x2(conv1)  # 14 14 1

    conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64])
    conv2_pool = max_pool_2x2(conv2)  # 7 7 1

    conv2_float = tf.reshape(conv2_pool, [-1, 7*7*64])
    full_1 = tf.nn.relu(full_layer(conv2_float, 1024))

    keep_prob = tf.placeholder(tf.float32)
    full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

    y_conv = full_layer(full1_drop, 10)

    train = optimizer(logit=y_conv, label=y, lr=LEARNING_LATE)
    accuary = test_accuracy(pred=y_conv, label=y)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(NUM_STEPS):
            batch = data.train.next_batch(MINIBATCH_SIZE)

            if step % 100 == 0:
                train_accuracy = sess.run(accuary, feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
                print("step {} train accuracy : {:0.4f}".format(step, train_accuracy))

            sess.run(train, feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

        X = data.test.images.reshape(10, 1000, 784)
        Y = data.test.labels.reshape(10, 1000, 10)
        test_acc = np.mean(list(sess.run(accuary,
                                         feed_dict={x: X[i], y: Y[i], keep_prob: 1.0 })
                                for i in range(10)))
        print("test accuarcy : {:0.4f}".format(test_acc))

if __name__ == "__main__":
    main()



