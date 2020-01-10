#!/usr/bin/python
# Author: Clara Vania

import numpy as np
import tensorflow as tf
import argparse
import time
import os
import _pickle as cPickle
import codecs
import glob
from utils import TextLoader
from word import WordLM
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, default='data/twitter/valid_clean.txt',
                        help="test file")
    parser.add_argument('--save_dir', type=str, default='model',
                        help='directory of the checkpointed models')
    parser.add_argument('--tokenizer_path', type=str, default='data/twitter/sp_20k.model',
                        help="path to sentencepiece tokenizer")
    parser.add_argument('--perplexity_file', type=str, 
            default='data/twitter/dstc6_t2_evaluation/hypotheses/hyp_ppl_10k.txt', help="perplexity output file")
    args = parser.parse_args()
    with codecs.open(args.perplexity_file, mode='w', encoding='utf-8') as wf:
        wf.truncate()
    test(args)


def run_epoch(session, m, data, data_loader, eval_op):
    costs = 0.0
    iters = 0
    state = m.initial_lm_state.eval()
    for step, (x, y) in enumerate(data_loader.data_iterator(data, m.batch_size, m.num_steps)):
        cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                     {m.input_data: x,
                                      m.targets: y,
                                      m.initial_lm_state: state})
        costs += cost
        iters += m.num_steps
    return np.exp(costs / iters)


def test(test_args):
    start = time.time()
    with open(os.path.join(test_args.save_dir, 'config.pkl'), 'rb') as f:
        args = cPickle.load(f)
    print(args)
    data_loader = TextLoader(args, train=False)
    file_list = glob.glob(f'{test_args.test_file}/x*')
    file_list.sort()
    # Model
    lm_model = WordLM
    args.word_vocab_size = data_loader.word_vocab_size
    print("Word vocab size: " + str(data_loader.word_vocab_size) + "\n")
    print("Begin testing...")
    # If using gpu:
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # gpu_config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
    # add parameters to the tf session -> tf.Session(config=gpu_config)
    with tf.Graph().as_default(), tf.Session() as sess:
        initializer = tf.random_uniform_initializer(-args.init_scale, args.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            mtest = lm_model(args, is_training=False, is_testing=True)

        # save only the last model
        saver = tf.train.Saver(tf.all_variables())
        tf.initialize_all_variables().run()
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for f in tqdm(file_list):
            test_data = data_loader.read_dataset(f)
            test_perplexity = run_epoch(sess, mtest, test_data, data_loader, tf.no_op())
            # print("Test Perplexity: %.3f" % test_perplexity)
            with codecs.open(test_args.perplexity_file, mode='a', encoding='utf-8') as wf: 
                wf.write(str(test_perplexity) + '\n')
        print("Test time: %.0f" % (time.time() - start))

if __name__ == '__main__':
    main()
