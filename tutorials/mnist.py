import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os


def check_dir(path=None):
    if not os.path.isdir(path):
        os.mkdir(path=path)


def get_placeholder(shape=None):
    return tf.placeholder(dtype=tf.float32, shape=shape)


def get_weight(shape=None):
    return tf.Variable(tf.random_normal(shape=shape, stddev=0.5))
    #return tf.Variable(tf.zeros(shape=shape))


def get_predict(x, w):
    return tf.matmul(x, w)


def get_cross_entropy(pred, label):
    logit = tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=label)
    cross_entropy = tf.reduce_mean(logit)
    return cross_entropy


def get_optimizer(learning_rate, logit):
    return tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(logit)
    # return tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(logit)


def get_accuracy(predict, label):
    correct_mask = tf.equal(tf.argmax(predict, 1), tf.argmax(label, 1))
    accuracy_percent = tf.reduce_mean(tf.cast(correct_mask, tf.float32))
    return accuracy_percent


def main():
    IMAGE_SIZE = 784
    LABEL_SIZE = 10
    LEARNING_LATE = 0.001

    DATA_DIR = r'../data/mnist'
    NUM_STEPS = 1000
    MINIBATCH_SIZE = 100

    def get_data(DATA_DIR):
        check_dir(path=DATA_DIR)
        return input_data.read_data_sets(train_dir=DATA_DIR, one_hot=True)

    def train_graph():
        tensor_value = dict()
        tensor_value['x'] = get_placeholder(shape=[None, IMAGE_SIZE])
        tensor_value['w'] = get_weight(shape=[IMAGE_SIZE, 20])
        tensor_value['w2'] = get_weight(shape=[20, IMAGE_SIZE])
        tensor_value['w3'] = get_weight(shape=[IMAGE_SIZE, LABEL_SIZE])
        tensor_value['y'] = get_placeholder(shape=[None, LABEL_SIZE])
        tensor_value['pred'] = get_predict(x=tensor_value['x'],
                                           w=tensor_value['w'])
        tensor_value['pred2'] = get_predict(x=tensor_value['pred'],
                                            w=tensor_value['w2'])
        tensor_value['pred3'] = get_predict(x=tensor_value['pred2'],
                                            w=tensor_value['w3'])
        return tensor_value

    def to_feed_dict(x_val, x_data, y_val, y_data):
        return {x_val: x_data, y_val: y_data}

    tensor = train_graph()
    logit = get_cross_entropy(pred=tensor['pred3'], label=tensor['y'])
    optimizer = get_optimizer(learning_rate=LEARNING_LATE, logit=logit)
    data = get_data(DATA_DIR=DATA_DIR)
    accuracy = get_accuracy(predict=tensor['pred3'], label=tensor['y'])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for _ in range(150):

            for _ in range(NUM_STEPS):
                batch_data, batch_label = data.train.next_batch(MINIBATCH_SIZE)
                sess.run(optimizer, feed_dict=to_feed_dict(tensor['x'], batch_data,
                                                           tensor['y'], batch_label))

            ans = sess.run(accuracy, feed_dict=to_feed_dict(tensor['x'], data.test.images,
                                                            tensor['y'], data.test.labels))
            print('Accuracy: {:.4}%'.format(ans))
    print('Accuracy: {:.4}%'.format(ans))


if __name__ == '__main__':
    main()