import tensorflow as tf
import numpy as np


def get_placeholder(shape):
    return tf.placeholder(dtype=tf.float32, shape=shape)


def get_variable(shape, name):
    return tf.Variable(tf.zeros(shape), dtype=tf.float32, name=name)


def get_pred(x, w, b):
    return tf.matmul(w, tf.transpose(x)) + b


def get_loss(label, pred):
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=pred)


def get_MSE(label, pred):
    return tf.reduce_mean(tf.square(label - pred))


def get_gradient_descent(lr, loss):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    train = optimizer.minimize(loss)
    return train


def main():
    NUM_STEP = 10

    def get_random_data() -> dict:
        w_real = [0.3, 0.5, 0.1]
        b_real = -0.2
        noise = np.random.randn(1, 2000) * 0.1

        data_dict = dict()
        data_dict['x_data'] = np.random.randn(2000, 3)
        data_dict['y_data'] = np.matmul(w_real, data_dict['x_data'].T) + b_real + noise
        return data_dict

    data = get_random_data()
    g = tf.Graph()
    with g.as_default():
        with tf.name_scope('placeholder'):
            x = get_placeholder(shape=[None, 3])
            y_label = get_placeholder(shape=None)

        with tf.name_scope('inference'):
            w = get_variable(shape=[1, 3], name='weight')
            b = get_variable(shape=1, name='bias')
            y_pred = get_pred(x=x, w=w, b=b)

        with tf.name_scope('loss'):
            loss = get_MSE(label=y_label, pred=y_pred)

        with tf.name_scope('train'):
            train = get_gradient_descent(lr=0.5, loss=loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for step in range(NUM_STEP+1):
                sess.run(train, feed_dict={x: data['x_data'], y_label: data['y_data']})

                if (step % 5 == 0):
                    print(step, sess.run([w, b]))

if __name__ == '__main__':
    main()