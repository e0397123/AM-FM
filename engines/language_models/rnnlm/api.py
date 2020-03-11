from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from .main import FLAGS, define_graph
import sentencepiece as spm
import codecs
import pickle
import os


class EncodeModel:
    @staticmethod
    def ini_flag(kwargs=None):
        FLAGS.set_default("model_path",
                          "../../../examples/dstc6/pretrain/lstm_LM/")
        FLAGS.set_default("tokenizer_path",
                          "../../../examples/dstc6/pretrain/sentencepiece/twitter_10000.model")
        FLAGS.set_default("bpe_vocab",
                          "../../../examples/dstc6/pretrain/sentencepiece/twitter_10000.vocab")

        FLAGS.set_default("embedding_name", "embeddings_full.npy")
        FLAGS.set_default("num_nodes", "150,105,70")
        FLAGS.set_default("use_sp", True)
        FLAGS.mark_as_parsed()

        if kwargs:
            for k in kwargs.keys():
                FLAGS.set_default(k, kwargs[k])

    def __init__(self, is_training=False, kwargs=None):
        assert not is_training

        EncodeModel.ini_flag(kwargs)
        self.flags = FLAGS.flag_values_dict()

        self.input_inputs, self.input_labels, self.perplexity_without_exp = define_graph(return_key="for_eval")

        # load best model
        self.sess = tf.compat.v1.Session()
        saver = tf.compat.v1.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint(self.flags["model_path"]))

        if self.flags["use_sp"]:
            sp = spm.SentencePieceProcessor()
            sp.Load(self.flags["tokenizer_path"])
            self.tokenizer = sp.EncodeAsPieces

            with codecs.open(self.flags["bpe_vocab"], mode='r', encoding='utf-8') as rf:
                lines = rf.readlines()
            lines = [line.strip().split('\t')[0] for line in lines]
            self.dictionary = {v: k for k, v in enumerate(lines)}
            self.reverse_dictionary = {k: v for k, v in enumerate(lines)}
        else:
            self.tokenizer = lambda x: x.strip().split()

            self.dictionary = pickle.load(open(os.path.join(self.flags["model_path"], 'word2id.dict'), 'rb'))
            self.reverse_dictionary = {k: v for k, v in self.dictionary.items()}

    def prepare_line(self, sent):
        line = ['<s>', '</s>']
        line[1:1] = self.tokenizer(sent)

        return line

    def test(self):
        a = "hello"
        b = "how are you"
        c = "good morning"

        output = self.encode([a, b, c])
        print(output)
        print("Done")

    def encode(self, sentences):

        all_input_ids = self.convert_sentences_to_features(sentences)

        perplexity_output = []
        for input_ids in all_input_ids:
            inputs = input_ids[:-1]
            labels = input_ids[1:]
            assert len(inputs) == len(labels)
            sent_length = len(inputs)

            sent_perplexity = 0
            for j in range(sent_length):
                # Run validation phase related TensorFlow operations
                t_perp = self.sess.run(
                    self.perplexity_without_exp,
                    feed_dict={self.input_inputs: [inputs[j] * 1.0],
                               self.input_labels: [labels[j] * 1.0]}
                )
                sent_perplexity += t_perp
            t_perplexity = np.exp(sent_perplexity / sent_length)
            perplexity_output.append(t_perplexity)

        return np.asarray(perplexity_output)

    def convert_sentences_to_features(self, sentences):
        # load hypothesis and references

        all_input_ids = []
        for sent in sentences:
            tokenized_line = self.prepare_line(sent.strip())
            token_ids = self.convert_tokens_to_ids(tokenized_line)

            all_input_ids.append(token_ids)

        return all_input_ids

    def convert_tokens_to_ids(self, tokens):
        return [self.dictionary.get(token, self.dictionary['<unk>']) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.reverse_dictionary.get(idx, '<unk>') for idx in ids]


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity("DEBUG")
    bert = EncodeModel()
    bert.test()
