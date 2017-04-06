"""Preprocesses Cornell Movie Dialog data."""
import nltk
import random
import tensorflow as tf

tf.app.flags.DEFINE_string("raw_data", "../corpus/movie_lines_utf8.txt", "")
tf.app.flags.DEFINE_string("out_train_file", "../corpus/movie_train.txt", "")
tf.app.flags.DEFINE_string("out_val_file", "../corpus/movie_val.txt", "")
tf.app.flags.DEFINE_string("out_test_file", "../corpus/movie_test.txt", "")

FLAGS = tf.app.flags.FLAGS

TRAINING_RATE = 0.1
VAL_RATE = 0.1
TEST_RATE = 0.1

def main(_):
    with open(FLAGS.raw_data, "r", encoding='utf8') as raw_data, \
        open(FLAGS.out_train_file, "w", encoding='utf8') as out_train, \
        open(FLAGS.out_val_file, "w", encoding='utf8') as out_val, \
        open(FLAGS.out_test_file, "w", encoding='utf8') as out_test:

        print('Processing...')
        preprocessed = []

        for line in raw_data:
            parts = line.split(" +++$+++ ")
            dialog_line = parts[-1]
            s = dialog_line.strip().lower()
            preprocessed.append(" ".join(nltk.word_tokenize(s)))

        size = len(preprocessed)
        train_size, val_size, test_size = int(size * TRAINING_RATE), int(size * VAL_RATE), int(size * TEST_RATE)
        print('Total lines: {0}\nTrain lines: {1}\nValidation lines: {2}\nTest lines: {3}.'.format(size, train_size, val_size, test_size))

        random.shuffle(preprocessed)

        for index, line in enumerate(preprocessed):
            if index < train_size:
                out_train.write(line + '\n')
            elif index < train_size + val_size:
                out_val.write(line + '\n')
            elif index < train_size + val_size + test_size:
                out_test.write(line + '\n')
            else:
                break

        print('Done.')
if __name__ == "__main__":
    tf.app.run()
