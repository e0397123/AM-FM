import torch
import collections
import numpy as np
import os
import logging
import tensorflow as tf
import codecs
import pickle
import sentencepiece as spm
from tqdm import tqdm
from tensorflow.contrib import rnn
from six.moves import range
from fairseq.models.roberta import XLMRModel

flags = tf.app.flags

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

# load sentencepiece tokenizer

flags.DEFINE_string('data_path', ".", "data directory.")
flags.DEFINE_string('model_path', "models/lstm-lm", "path to which model is saved.")
flags.DEFINE_string('pretrained_model', "FM/xlmr.base", "path to pretrained xlm model.")
flags.DEFINE_string('feature_path', "FM/features", "path to store extracted features.")
flags.DEFINE_string('tokenizer_path', "data/bpe_full.model", "path to sentencepiece tokenizer")
flags.DEFINE_integer('batch_size', 256, "number of samples in each batch.")
flags.DEFINE_integer('seq_len', 50, "fix the sequence length")
flags.DEFINE_integer('embedding_size', 128, 'specify embedding dimension')
flags.DEFINE_integer('decay_threshold', 5, 'learning rate decay after decay_threshold epochs')
flags.DEFINE_integer('valid_summary', 1, 'validation interval')
flags.DEFINE_list('num_nodes', '150,75', 'hidden layer size')
flags.DEFINE_string('save_path', "fm.prob", "path to save sentence-level probs")
flags.DEFINE_float('dropout', 0.2, 'dropout rate')

FLAGS = flags.FLAGS

xlmr = XLMRModel.from_pretrained(FLAGS.pretrained_model, checkpoint_file='model.pt')
xlmr.eval()

# load sentencepiece tokenizer
sp = spm.SentencePieceProcessor()
sp.Load(FLAGS.tokenizer_path)
vocabulary_size = sp.GetPieceSize()

# function for reading data from each input file
def read_single_case(line):
    tokenized_line = sp.EncodeAsIds(line.strip())
    tokenized_line.insert(0, sp.bos_id())
    if len(tokenized_line) > FLAGS.seq_len:
        tokenized_line = tokenized_line[:FLAGS.seq_len]
        tokenized_line.append(sp.eos_id())
    else:
        for i in range(len(tokenized_line), FLAGS.seq_len, 1):
            tokenized_line.append(sp.eos_id())
        tokenized_line.append(sp.eos_id())
    return tokenized_line

# function for reading data from each input file
def read_data(filename):
    data = []
    with codecs.open(filename, encoding='utf-8', mode='r') as f:
        lines = f.readlines()

    for line in tqdm(lines):
        if line.strip():
            data.extend(read_single_case(line))
    return data



class DataGeneratorSeq(object):

    def __init__(self, text, batch_size, num_unroll):
        self._text = text
        self._text_size = len(self._text)
        # Number of datapoints in a batch of data
        self._batch_size = batch_size
        # Num unroll is the number of steps we unroll the RNN in a single training step
        self._num_unroll = num_unroll
        self._segments = self._text_size // self._batch_size

        self._cursor = [offset * self._segments for offset in range(self._batch_size)]

    def next_batch(self):
        '''
        Generates a single batch of data
        '''
        # Train inputs (one-hot-encoded) and train outputs (one-hot-encoded)
        batch_data = np.zeros((self._batch_size), dtype=np.float32)
        batch_labels = np.zeros((self._batch_size), dtype=np.float32)

        # Fill in the batch datapoint by datapoint
        for b in range(self._batch_size):
            # If the cursor of a given segment exceeds the segment length
            # we reset the cursor back to the beginning of that segment
            if self._cursor[b] + 1 >= self._text_size:
                self._cursor[b] = b * self._segments

            # Add the text at the cursor as the input
            batch_data[b] = self._text[self._cursor[b]]
            # Add the preceding word as the label to be predicted
            batch_labels[b] = self._text[self._cursor[b] + 1]
            # Update the cursor
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size

        return batch_data, batch_labels

    def unroll_batches(self):
        '''
        This produces a list of num_unroll batches
        as required by a single step of training of the RNN
        '''
        unroll_data, unroll_labels = [], []
        for ui in range(self._num_unroll):
            data, labels = self.next_batch()
            unroll_data.append(data)
            unroll_labels.append(labels)

        return unroll_data, unroll_labels

    def reset_indices(self):
        '''
        Used to reset all the cursors if needed
        '''
        self._cursor = [offset * self._segments for offset in range(self._batch_size)]

    def get_segment_size(self):
        return self._segments // self._num_unroll


if __name__ == "__main__":

    # ## Reading data
    logging.info('Processing file %s' % FLAGS.data_path)
    words = read_data(FLAGS.data_path)
    logging.info('Data size (Characters) %d' % (len(words)))
    logging.info('Sample string %s' % (words[:50]))

    # =========================================================
    # Define Graph

    # Number of neurons in the hidden state variables
    num_nodes = [int(node) for node in FLAGS.num_nodes]

    # ## Defining Inputs and Outputs

    tf.reset_default_graph()

    # Training Input data.

    # Defining unrolled training inputs
    train_inputs = tf.placeholder(tf.float32, shape=[FLAGS.seq_len, FLAGS.batch_size, 768], name='train_inputs')
    train_labels = tf.placeholder(tf.int32, shape=[FLAGS.seq_len, FLAGS.batch_size], name='train_labels')
    train_labels_ohe = tf.one_hot(train_labels, vocabulary_size)

    # Validation data placeholders
    valid_inputs = tf.placeholder(tf.float32, shape=[FLAGS.seq_len, 1, 768], name='valid_inputs')
    valid_labels = tf.placeholder(tf.int32, shape=[FLAGS.seq_len, 1], name='valid_labels')
    valid_labels_ohe = tf.one_hot(valid_labels, vocabulary_size)

    # downsample the embedding
    with tf.variable_scope("down_sample_layer", reuse=tf.AUTO_REUSE):
        # [seq_len, batch_size, 300]
        train_embeds = tf.layers.dense(train_inputs, units=FLAGS.embedding_size,
                                              kernel_initializer=tf.keras.initializers.glorot_normal())

    with tf.variable_scope("down_sample_layer", reuse=tf.AUTO_REUSE):
        # [seq_len, 1, 300]
        valid_embeds = tf.layers.dense(valid_inputs, units=FLAGS.embedding_size,
                                              kernel_initializer=tf.keras.initializers.glorot_normal())


    # Defining embedding lookup operations for all the unrolled
    # trianing inputs

    # ## Defining Model Parameters

    logging.info('Defining softmax weights and biases')
    # Softmax Classifier weights and biases.
    w = tf.Variable(tf.truncated_normal([num_nodes[-1], vocabulary_size], stddev=0.01))
    b = tf.Variable(tf.random_uniform([vocabulary_size], 0.0, 0.01))

    logging.info('Defining the LSTM cell')
    # Defining a deep LSTM from Tensorflow RNN API

    # First we define a list of LSTM cells
    # num_nodes here is a sequence of hidden layer sizes
    cells = [tf.nn.rnn_cell.LSTMCell(n) for n in num_nodes]

    # We now define a dropout wrapper for each LSTM cell
    dropout_cells = [
        rnn.DropoutWrapper(
            cell=lstm, input_keep_prob=1.0,
            output_keep_prob=1.0 - FLAGS.dropout, state_keep_prob=1.0,
            variational_recurrent=True,
            input_size=tf.TensorShape([FLAGS.embedding_size]),
            dtype=tf.float32
        ) for lstm in cells
    ]

    # We first define a MultiRNNCell Object that uses the
    # Dropout wrapper (for training)
    stacked_dropout_cell = tf.nn.rnn_cell.MultiRNNCell(dropout_cells)
    # Here we define a MultiRNNCell that does not use dropout
    # Validation and Testing
    stacked_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

    # ## Defining LSTM Computations

    logging.info('LSTM calculations for unrolled inputs and outputs')
    # =========================================================
    # Training inference logic

    # Initial state of the LSTM memory.
    initial_state = stacked_dropout_cell.zero_state(FLAGS.batch_size, dtype=tf.float32)

    # Defining the LSTM cell computations (training)
    train_outputs, initial_state = tf.nn.dynamic_rnn(
        stacked_dropout_cell, train_embeds,
        time_major=True, initial_state=initial_state
    )

    # Reshape the final outputs to [seq_len*batch_size, num_nodes]
    final_output = tf.reshape(train_outputs, [-1, num_nodes[-1]])

    # Computing logits
    logits = tf.matmul(final_output, w) + b
    # Computing predictions
    train_prediction = tf.nn.softmax(logits)

    # Reshape logits to time-major fashion [seq_len, batch_size, vocabulary_size]
    time_major_train_logits = tf.reshape(logits, [FLAGS.seq_len, FLAGS.batch_size, -1])
    train_prediction = tf.reshape(train_prediction, [FLAGS.seq_len, FLAGS.batch_size, -1])

    # Perplexity related operation
    train_perplexity_without_exp = tf.reduce_sum(train_labels_ohe * -tf.log(train_prediction + 1e-10)) / (
                FLAGS.seq_len * FLAGS.batch_size)

    # ## Calculating LSTM Loss

    loss = tf.contrib.seq2seq.sequence_loss(
        logits=tf.transpose(time_major_train_logits, [1, 0, 2]),
        targets=tf.transpose(train_labels),
        weights=tf.ones([FLAGS.batch_size, FLAGS.seq_len], dtype=tf.float32),
        average_across_timesteps=False,
        average_across_batch=True
    )

    loss = tf.reduce_sum(loss)

    # =========================================================
    # Validation inference logic

    # Separate state for validation data
    initial_valid_state = stacked_cell.zero_state(1, dtype=tf.float32)

    # Validation input related LSTM computation
    valid_outputs, initial_valid_state = tf.nn.dynamic_rnn(
        stacked_cell, valid_embeds,
        time_major=True, initial_state=initial_valid_state
    )

    # Reshape the final outputs to [seq_len, num_nodes]
    final_valid_output = tf.reshape(valid_outputs, [-1, num_nodes[-1]])
    final_valid_labels_ohe = tf.reshape(valid_labels_ohe, [-1, vocabulary_size])

    # Computing logits
    valid_logits = tf.matmul(final_valid_output, w) + b
    # Computing predictions
    valid_prediction = tf.nn.softmax(valid_logits)

    # Perplexity related operation
    valid_perplexity_without_exp = tf.reduce_mean(tf.reduce_sum(final_valid_labels_ohe *
                                                                -tf.log(valid_prediction + 1e-10), axis=1))

    # ### Running Training, Validation and Generation
    valid_perplexity_ot = []

    session = tf.InteractiveSession()
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()
    logging.info('Initialized')

    test_gen = DataGeneratorSeq(words, 1, FLAGS.seq_len)
    test_steps_per_document = test_gen.get_segment_size()

    feed_dict = {}
    test_loss = 0
    # =========================================================
    # Training Procedure

    logging.info('Restoring model from {}'.format(FLAGS.model_path))
    # load best model
    saver.restore(session, tf.train.latest_checkpoint(FLAGS.model_path))
    
    with codecs.open(FLAGS.save_path, encoding='utf-8', mode='w') as wf:
        wf.truncate()

    for t_step in tqdm(range(test_steps_per_document)):
        utest_data, utest_labels = test_gen.unroll_batches()
        # [batch_size, seq_len, 1024]
        utest_data_tensor = torch.from_numpy(np.transpose(np.array(utest_data, dtype=np.int64)))
        utest_last_layer_features = xlmr.extract_features(utest_data_tensor).detach().numpy()
        utest_last_layer_features = np.transpose(utest_last_layer_features, [1, 0, 2])
        
        # Run validation phase related TensorFlow operations
        t_perp = session.run(
            valid_perplexity_without_exp,
            feed_dict={valid_inputs: utest_last_layer_features, valid_labels: np.array(utest_labels)}
        )

        test_loss += t_perp
        with codecs.open(FLAGS.save_path, encoding='utf-8', mode='a') as wf:
            wf.write(str(np.exp(t_perp)) + '\n')

    logging.info('')
    t_perplexity = np.exp(test_loss / test_steps_per_document)
    logging.info("test Perplexity: %.2f\n" % t_perplexity)

    #     with codecs.open(FLAGS.hyp_out, mode='w') as f:
    #         f.truncate()
    #     with codecs.open(FLAGS.hyp_out, mode='a') as f:
    #         for item in hyp_perplexity_ot:
    #             f.write(str(item) + '\n')
    #
    #     logging.info('Done writing hypothesis perplexity file to {}'.format(FLAGS.hyp_out))

    session.close()


