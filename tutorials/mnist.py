import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


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


def optimizer(learning_rate, logit):
    return tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(logit)


def accuracy(predict, label):
    correct_mask = tf.equal(tf.argmax(predict, 1), tf.argmax(label, 1))
    accuracy_percent = tf.reduce_mean(correct_mask, tf.float32)
    return accuracy_percent


def main():
    IMAGE_SIZE = 784
    LABEL_SIZE = 10
    LEARNING_LATE = 0.1
    DATA_DIR = '../data/mnist'
    NUM_STEPS = 1000
    MINIBATCH_SIZE = 100

    def get_data(DATA_DIR):
        return input_data.read_data_sets(train_dir=DATA_DIR, one_hot=True)

    def train_graph():
        tensor_value = dict()
        tensor_value['x'] = get_placeholder(shape=[None, IMAGE_SIZE])
        tensor_value['w'] = get_weight(shape=[IMAGE_SIZE, LABEL_SIZE])
        tensor_value['y'] = get_placeholder(shape=[None, LABEL_SIZE])
        tensor_value['pred'] = get_predict(x=tensor_value['x'],
                                           w=tensor_value['w'])
        return tensor_value

    tensor = train_graph()
    data = get_data(DATA_DIR=DATA_DIR)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for _ in range(NUM_STEPS):
            batch_data, batch_label = data.train.next_batch(MINIBATCH_SIZE)
