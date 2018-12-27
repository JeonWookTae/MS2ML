import tensorflow as tf
import numpy as np


def get_sentence_data():
    with open(f'../data/sentence/test.txt') as f:
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
    index = np.random.choice(range(len(data)), size, replace=False)
    inputs = [data[i][0] for i in index]
    labels = [data[i][1] for i in index]
    return inputs, labels

