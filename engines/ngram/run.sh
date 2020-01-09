#!/bin/bash

vocab_size=100k

python create_vocab.py \
	--train_file ../../data/twitter/train_clean_${vocab_size}.txt \
	--vocab_file twitter_${vocab_size}.vocab


ngram-count -vocab twitter_${vocab_size}.vocab \
	-text ../../data/twitter/train_clean_${vocab_size}.txt \
	-order 5 \
	-write twitter_${vocab_size}.count \
	-unk

ngram-count \
	-vocab twitter_${vocab_size}.count \
	-read twitter_${vocab_size}.count \
	-order 5 \
	-lm twitter_${vocab_size}.lm \
	-kndiscount1 \
	-kndiscount2 \
	-kndiscount3 \
	-kndiscount4 \
	-kndiscount5

ngram \
	-ppl ../../data/twitter/valid_clean.txt \
	-order 1 \
	-lm twitter_${vocab_size}.lm

ngram \
	-ppl ../../data/twitter/valid_clean.txt \
	-order 2 \
	-lm twitter_${vocab_size}.lm

ngram \
	-ppl ../../data/twitter/valid_clean.txt \
	-order 3 \
	-lm twitter_${vocab_size}.lm

ngram \
	-ppl ../../data/twitter/valid_clean.txt \
	-order 4 \
	-lm twitter_${vocab_size}.lm

ngram \
	-ppl ../../data/twitter/valid_clean.txt \
	-order 5 \
	-lm twitter_${vocab_size}.lm
