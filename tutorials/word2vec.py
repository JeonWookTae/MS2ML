import tensorflow as tf
import numpy as np


def get_sentence_data(path):
    with open(path) as f:
        data = f.read()
    return data


def get_sequence(sentence):
    return sentence.split()


def get_word_dict(sentence):
    word_list = get_sequence(sentence=sentence)
    word_list = list(set(word_list))
    word_dict = {w: i for i, w in enumerate(word_list)}
    return word_dict


def get_skip_gram(sentence, word_dict):
    seq = sentence.split()
    seq_len = len(seq) - 1
    words_list_a = [[word_dict[seq[index]], word_dict[seq[index - 1]]] for index in range(1, seq_len)]
    words_list_b = [[word_dict[seq[index]], word_dict[seq[index + 1]]] for index in range(1, seq_len)]
    skip_gram = words_list_a + words_list_b
    return skip_gram


def get_batch(data, size):
    index = np.random.choice([i for i in range(len(data))], size, replace=False)
    inputs = [data[i][0] for i in index]
    labels = [[data[i][1]] for i in index]
    return inputs, labels


def get_variable(shape):
    init_variable = tf.random_uniform(shape=shape, minval=-1.0, maxval=1.0)
    return tf.Variable(init_variable)


def get_bias(shape):
    return tf.Variable(tf.zeros(shape=shape))


def get_word_skip_data(data):
    word_dict = get_word_dict(data)
    skip_gram = get_skip_gram(data, word_dict)
    return word_dict, skip_gram


def NCE_optimizer(nce_weight,
                  nce_bias,
                  label,
                  select_embedding,
                  sample,
                  voca_size,
                  lr):
    nce_loss = tf.nn.nce_loss(nce_weight, nce_bias, label, select_embedding, sample, voca_size)
    nce_loss_mean = tf.reduce_mean(nce_loss)
    optimizer = tf.train.AdamOptimizer(lr)
    train = optimizer.minimize(nce_loss_mean)
    return train, nce_loss_mean


def main():
    DATA_PATH = f'../data/sentence/test.txt'
    BATCH_SIZE = 20
    LEARNING_LATE = 0.001
    EMBEDDING_SIZE = 5
    SAMPLED = 15
    EPOCH = 200

    data = get_sentence_data(path=DATA_PATH)
    word_dict, skip_gram = get_word_skip_data(data)
    VOCA_SIZE = len(word_dict)

    inputs = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
    labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1])

    embeddings = get_variable(shape=[VOCA_SIZE, EMBEDDING_SIZE])
    select_embd = tf.nn.embedding_lookup(embeddings, inputs)

    nce_weights = get_variable(shape=[VOCA_SIZE, EMBEDDING_SIZE])
    nce_biases = get_bias(shape=[VOCA_SIZE])

    train, loss = NCE_optimizer(nce_weight=nce_weights,
                                nce_bias=nce_biases,
                                label=labels,
                                select_embedding=select_embd,
                                sample=SAMPLED,
                                voca_size=VOCA_SIZE,
                                lr=LEARNING_LATE)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(EPOCH + 1):
            batch_inputs, batch_labels = get_batch(skip_gram, BATCH_SIZE)

            feed_dict = {inputs: batch_inputs, labels: batch_labels}
            _, loss_val = sess.run([train, loss], feed_dict=feed_dict)

            if step % 10 == 0:
                print("step: {} by loss: {}".format(step, loss_val))


if __name__ == "__main__":
    main()
