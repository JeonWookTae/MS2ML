import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

setence_num = 10000
batch_size = 64
embedding_dimension = 5
negative_samples = 8
LOG_DIR = r'../data/logs/word2vec_intro'
digit_to_word_map = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
                     6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}


def make_sentence(sentence_num) -> list:
    sentence_list = list()
    for i in range(sentence_num):
        rand_odd_init = np.random.choice(range(1, 10, 2), 3)
        sentence_list.append(" ".join([digit_to_word_map[index] for index in rand_odd_init]))
        rand_even_init = np.random.choice(range(2, 10, 2), 3)
        sentence_list.append(" ".join([digit_to_word_map[index] for index in rand_even_init]))
    return sentence_list


def make_word_map(sentence) -> (dict, dict):
    word2index_map = dict()
    index = 0
    for word_list in (sent.lower().split() for sent in sentence):
        for word in word_list:
            if word not in word2index_map:
                word2index_map.update({word: index})
                index += 1
    index2word_map = {index: word for word, index in word2index_map.items()}
    return index2word_map, word2index_map


def make_skip_gram(sentence, word2index_map) -> list:
    skip_gram_pairs_list = list()
    token_side_pair = lambda i: [word2index_map[token_sentence[i - 1]], word2index_map[token_sentence[i + 1]]]
    token_pair_lambda = lambda i: [token_side_pair(i), word2index_map[token_sentence[i]]]

    for token_sentence in (sent.lower().split() for sent in sentence):
        token_range = range(1, len(token_sentence) - 1)
        token_pair_list = [token_pair_lambda(i) for i in token_range]
        skip_gram_pairs_list.extend([[token[1], token[0][0]] for token in token_pair_list])
        skip_gram_pairs_list.extend([[token[1], token[0][1]] for token in token_pair_list])
    return skip_gram_pairs_list


def get_skip_batch(batch_size, skip_gram_paris):
    instance_indices = list(range(len(skip_gram_paris)))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    x = [skip_gram_paris[i][0] for i in batch]
    y = [[skip_gram_paris[i][1]] for i in batch]
    return x, y


sentence_list = make_sentence(sentence_num=setence_num)
index2word_map, word2index_map = make_word_map(sentence=sentence_list)
skip_gram_pairs = make_skip_gram(sentence=sentence_list, word2index_map=word2index_map)
vocabulary_size = len(index2word_map)

train_inputs = tf.placeholder(shape=[batch_size], dtype=tf.int32)
train_labels = tf.placeholder(shape=[batch_size, 1], dtype=tf.int32)

with tf.variable_scope('embeddings'):
    embeddings = tf.Variable(tf.truncated_normal(shape=[vocabulary_size, embedding_dimension], mean=-1.0, stddev=1.0))
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

with tf.variable_scope('nce_variable'):
    nce_weight = tf.Variable(tf.truncated_normal(shape=[vocabulary_size, embedding_dimension],
                                                 stddev=1.0 / math.sqrt(embedding_dimension)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

with tf.name_scope('nce_train'):
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weight, biases=nce_biases, inputs=embed,
                       labels=train_labels, num_sampled=negative_samples, num_classes=vocabulary_size)
    )
    global_step = tf.Variable(0, trainable=False)
    learningRate = tf.train.exponential_decay(learning_rate=0.1,
                                              global_step=global_step,
                                              decay_steps=1000,
                                              decay_rate=0.95,
                                              staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate=learningRate).minimize(loss)
    tf.summary.scalar('loss', loss)

# 모든 요약 연산을 병합
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(LOG_DIR, graph=tf.get_default_graph())
    saver = tf.train.Saver()

    with open(os.path.join(LOG_DIR, 'metadata.tsv'), 'w') as metadata:
        metadata.write('Name\tClass\n')
        for k, v in index2word_map.items():
            metadata.write('%s\t%d\n' % (v, k))

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embeddings.name

    # 임베딩을 메타데이터 파일과 연결
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(train_writer, config)

    for step in range(1000):
        x_batch, y_batch = get_skip_batch(batch_size=batch_size, skip_gram_paris=skip_gram_pairs)
        feed_dict = {train_inputs: x_batch, train_labels: y_batch}
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)

        if step % 100 == 0:
            saver.save(sess, os.path.join(LOG_DIR, "w2v_model.ckpt"), step)
            feed_dict = {train_inputs: x_batch, train_labels: y_batch}
            loss_value = sess.run(loss, feed_dict=feed_dict)
            print("loss at %d: %.5f" % (step, loss_value))

        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        normalized_embeddings_matrix = sess.run(normalized_embeddings)
