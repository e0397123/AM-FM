# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract pre-computed feature vectors from BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import tokenization
# import tokenization
import tensorflow as tf
import numpy as np
from numpy import linalg

# from modeling import BertModel, BertConfig
from .modeling import BertModel, BertConfig
# from extract_features import FLAGS, InputFeatures
from .extract_features import FLAGS, InputFeatures


class EncodeModel:
    @staticmethod
    def ini_flag(kwargs=None):
        FLAGS.set_default("bert_config_file",
                          "/media/datadrive/PycharmProject/Singapore/A_F_PM/model_pretrain/bert/"
                          "uncased_L-12_H-768_A-12/bert_config.json")
        FLAGS.set_default("vocab_file",
                          "/media/datadrive/PycharmProject/Singapore/A_F_PM/model_pretrain/bert/"
                          "uncased_L-12_H-768_A-12/vocab.txt")
        FLAGS.set_default("init_checkpoint",
                          "/media/datadrive/PycharmProject/Singapore/A_F_PM/AM_FM_PM/examples/dstc6/pretrain/bert/")

        FLAGS.set_default("max_seq_length", 128)
        FLAGS.set_default("layers", "-1,-2,-3,-4")
        FLAGS.set_default("do_lower_case", True)
        FLAGS.set_default("use_one_hot_embeddings", False)

        if kwargs:
            for k in kwargs.keys():
                FLAGS.set_default(k, kwargs[k])

        FLAGS.mark_as_parsed()

    @staticmethod
    def strategy(embeddings_list, strategy='top-layer-embedding-average'):

        output_emb = []
        for embeddings in embeddings_list:
            obj_emb = None
            if strategy == 'top-layer-embedding-average':
                # embedding average
                obj_emb = embeddings["layer_output_0"]
                obj_emb = np.sum(obj_emb, axis=0)
                obj_emb = obj_emb / linalg.norm(obj_emb)

            output_emb.append(obj_emb)

        return np.asarray(output_emb)

    def __init__(self, is_training=False, kwargs=None):
        EncodeModel.ini_flag(kwargs)
        self.flags = FLAGS.flag_values_dict()

        self.layer_indexes = [int(x) for x in self.flags["layers"].split(",")]
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=self.flags["vocab_file"], do_lower_case=self.flags["do_lower_case"])

        bert_config = BertConfig.from_json_file(self.flags["bert_config_file"])
        self.input_ids = tf.compat.v1.placeholder(tf.int32,
                                                  shape=[None, self.flags["max_seq_length"]], name="input_ids")
        self.input_mask = tf.compat.v1.placeholder(tf.int32,
                                                   shape=[None, self.flags["max_seq_length"]], name="input_mask")
        self.segment_ids = tf.compat.v1.placeholder(tf.int32,
                                                    shape=[None, self.flags["max_seq_length"]], name="segment_ids")

        self.model = BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=self.flags["use_one_hot_embeddings"])

        self.sess = tf.compat.v1.Session()
        saver = tf.compat.v1.train.Saver()

        saver.restore(self.sess, tf.train.latest_checkpoint(self.flags["init_checkpoint"]))

    def test(self):
        a = "hello"
        b = "how are you"
        c = "good morning"

        output = self.encode([a, b, c])
        print(output)
        print("Done")

    def encode(self, sentences):
        assert isinstance(sentences, list)
        _, model_input = self.convert_sentences_to_features(sentences)

        all_layers = self.sess.run(self.model.all_encoder_layers,
                                   feed_dict={self.input_ids: model_input["input_ids"],
                                              self.input_mask: model_input["input_mask"],
                                              self.segment_ids: model_input["input_type_ids"]})

        sent_emb_list = []
        for sent_n in range(len(sentences)):
            specific_layer_output = {}
            sent_length = model_input["input_length"][sent_n]

            for (i, layer_index) in enumerate(self.layer_indexes):
                layer_output = all_layers[layer_index]
                sent_emb = layer_output[sent_n][:sent_length]
                specific_layer_output["layer_output_%d" % i] = sent_emb

            sent_emb_list.append(specific_layer_output)

        return EncodeModel.strategy(sent_emb_list)

    def convert_sentences_to_features(self, sentences):
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        all_unique_ids = []
        all_input_ids = []
        all_input_mask = []
        all_input_type_ids = []
        all_input_length = []
        for (ex_index, example) in enumerate(sentences):
            tokens_a = self.tokenizer.tokenize(example)

            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > self.flags["max_seq_length"] - 2:
                tokens_a = tokens_a[0:(self.flags["max_seq_length"] - 2)]

            tokens = ["[CLS]"]
            tokens.extend(tokens_a)
            tokens.append("[SEP]")
            seq_length = len(tokens)

            input_type_ids = [0] * seq_length
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * seq_length
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # Zero-pad up to the sequence length.
            pad_length = self.flags["max_seq_length"] - seq_length
            input_ids.extend([0] * pad_length)
            input_mask.extend([0] * pad_length)
            input_type_ids.extend([0] * pad_length)

            features.append(
                InputFeatures(
                    unique_id=ex_index,
                    tokens=tokens,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    input_type_ids=input_type_ids))

            all_unique_ids.append(ex_index)
            all_input_ids.append(input_ids)
            all_input_mask.append(input_mask)
            all_input_type_ids.append(input_type_ids)
            all_input_length.append(seq_length)

        model_input = {
            "input_ids": all_input_ids,
            "input_mask": all_input_mask,
            "input_type_ids": all_input_type_ids,
            "input_length": all_input_length
        }

        return features, model_input


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity("DEBUG")
    bert = EncodeModel()
    # bert.test()
