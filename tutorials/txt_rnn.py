import numpy as np
import tensorflow as tf
from collections import namedtuple

batch_size = 128
embedding_dimension = 64
num_classes = 2
hidden_layer = 32
times_steps = 6
data_num = 10000
num_LSTM_layer = 2
digit_to_word_map = {0: "PAD", 1: "one", 2: "two", 3: "three", 4: "four",
                     5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}


def make_seq_data(data_num):
    event_sentences = list()
    odd_sentence = list()
    seqlens = list()
    for _ in range(data_num):
        rand_seq_len = np.random.choice(range(3, 7))
        seqlens.append(rand_seq_len)
        # 홀수를 뽑자
        rand_odd_ints = np.random.choice(range(1, 10, 2), rand_seq_len)
        # 짝수를 뽑자
        rand_even_ints = np.random.choice(range(2, 10, 2), rand_seq_len)

        if rand_seq_len < 6:
            rand_odd_ints = np.append(rand_odd_ints, [0] * (6 - rand_seq_len))
            rand_even_ints = np.append(rand_even_ints, [0] * (6 - rand_seq_len))

        event_sentences.append(" ".join([digit_to_word_map[r] for r in rand_odd_ints]))
        odd_sentence.append(" ".join([digit_to_word_map[r] for r in rand_even_ints]))
    return event_sentences, odd_sentence, seqlens*2


def make_word_index(data):
    word2index_map = dict()
    index = 0
    for sent in data:
        for word in sent.lower().split():
            if word not in word2index_map:
                word2index_map[word] = index
                index += 1

    index2word_map = {index: word for word, index in word2index_map.items()}
    return word2index_map, len(index2word_map)


def make_data(data, seqlens, data_num):
    assert len(data) >= data_num, 'need to data length more then bigger data_num'
    labels = [1] * data_num + [0] * data_num
    for i in range(len(labels)):
        label = labels[i]
        one_hot_encoding = [0] * 2
        one_hot_encoding[label] = 1
        labels[i] = one_hot_encoding

    data_indices = list(range(len(data)))
    np.random.shuffle(data_indices)
    data = np.array(data)[data_indices]
    labels = np.array(labels)[data_indices]
    seqlens = np.array(seqlens)[data_indices]

    def train():
        train_x = data[:data_num]
        train_y = labels[:data_num]
        train_seqlens = seqlens[:data_num]
        return train_x, train_y, train_seqlens

    def test():
        test_x = data[data_num:]
        test_y = labels[data_num:]
        test_seqlens = seqlens[data_num:]
        return test_x, test_y, test_seqlens

    train_test_data = namedtuple('data', ['train', 'test'])
    train_test_data = train_test_data(train=train(), test=test())
    return train_test_data


def get_sentence_batch(batch_size, data_x, data_y, data_seqlens, word_index_map):
    instance_indices = list(range(len(data_x)))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    x = [[word_index_map[word] for word in data_x[i].lower().split()]
         for i in batch]
    y = [data_y[i] for i in batch]
    seqlens = [data_seqlens[i] for i in batch]
    return x, y, seqlens


event_sentences, odd_sentence, seqlens = make_seq_data(data_num=data_num)
data = event_sentences + odd_sentence
word2index_map, vocabulary_size = make_word_index(data=data)
train_x, train_y, train_seqlens = make_data(data=data, seqlens=seqlens, data_num=data_num).train
test_x, test_y, test_seqlens = make_data(data=data, seqlens=seqlens, data_num=data_num).test

with tf.name_scope("input_placeholder"):
    _inputs = tf.placeholder(tf.int32, shape=[batch_size, times_steps])
    _labels = tf.placeholder(tf.float32, shape=[batch_size, num_classes])
    _seqlens = tf.placeholder(tf.int32, shape=[batch_size])

with tf.name_scope("embedings"):
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_dimension], -1.0, 1.0), name="embedding")
    embed = tf.nn.embedding_lookup(embeddings, _inputs)

with tf.variable_scope("lstm"):
    lstm_cell_list = [tf.nn.rnn_cell.BasicLSTMCell(hidden_layer, forget_bias=1.0) for _ in range(num_LSTM_layer)]
    cell = tf.nn.rnn_cell.MultiRNNCell(cells=lstm_cell_list, state_is_tuple=True)
    outputs, states = tf.nn.dynamic_rnn(cell, embed, sequence_length=_seqlens, dtype=tf.float32)

with tf.variable_scope("weight"):
    weight = {"linear_layer": tf.Variable(tf.truncated_normal([hidden_layer, num_classes], mean=0, stddev=.01))}
    biases = {"linear_layer": tf.Variable(tf.truncated_normal([num_classes], mean=0, stddev=0.01))}

with tf.variable_scope("output"):
    final_output = tf.matmul(states[num_LSTM_layer - 1][1], weight["linear_layer"]) + biases["linear_layer"]

with tf.name_scope("softmax"):
    softmax = tf.nn.softmax_cross_entropy_with_logits_v2(logits=final_output, labels=_labels)
    cross_entropy = tf.reduce_mean(softmax)

with tf.name_scope("RMStrain"):
    train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(_labels, 1), tf.argmax(final_output, 1))
    accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32))) * 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1000):
        x_batch, y_batch, seqlen_batch = get_sentence_batch(batch_size, train_x, train_y, train_seqlens, word2index_map)
        feed_dict = {_inputs: x_batch, _labels: y_batch, _seqlens: seqlen_batch}
        sess.run(train_step, feed_dict=feed_dict)

        if step % 100 == 0:
            acc = sess.run(accuracy, feed_dict=feed_dict)
            print("accuracy at %d: %.5f" % (step, acc))

    for test_batch in range(5):
        x_test, y_test, seqlen_test = get_sentence_batch(batch_size, test_x, test_y, test_seqlens, word2index_map)
        feed_dict = {_inputs: x_test, _labels: y_test, _seqlens: seqlen_test}
        batch_pred, batch_acc = sess.run([tf.argmax(final_output, 1), accuracy], feed_dict=feed_dict)
        print("Test batch accuarcy %d: %.5f" % (test_batch, batch_acc))

    output_example = sess.run([outputs], feed_dict=feed_dict)
    state_example = sess.run([states[1]], feed_dict=feed_dict)
    print(output_example[0][1][:6, 0:3])
