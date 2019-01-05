import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def check_dir(path=None):
    if not os.path.isdir(path):
        os.mkdir(path=path)


def get_data(data_dir):
    check_dir(path=data_dir)
    return input_data.read_data_sets(train_dir=data_dir, one_hot=True)


LOG_DIR = "logs/RNN_with_summaries"
