#!/bin/bash

SIZE=10k

CUDA_VISIBLE_DEVICES=2 python main.py \
	--data_path=. \
	--dataset=twitter \
	--data_size=${SIZE} \
	--model_name=lstm-${SIZE}-whitespace \
	--embedding_name=embeddings.npy \
	--tokenizer_path=bpe_full.model \
	--hyp_out=ppl_hypothesis.txt \
	--ref_out=ppl_reference.txt \
	--batch_size=256 \
	--embedding_size=128 \
	--num_epochs=251 \
	--use_sp=False \
	--do_train=True \
	--do_eval=True \
	--do_dstc_eval=True
