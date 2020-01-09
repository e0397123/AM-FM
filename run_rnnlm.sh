#!/bin/bash

# step 1:
#python engines/rnnlm/create_raw_data.py \
#  --train_file data/twitter/train.txt \
#  --valid_file data/twitter/valid.txt \
#  --train_output data/twitter/train_rnnlm_100k.txt \
#  --valid_output data/twitter/valid_rnnlm.txt \
#  --data_size 10000
#

# step 2:
# spm_train --input=data/twitter/train_clean_1M.txt --model_prefix=data/twitter/sp_1m --vocab_size=32000 --character_coverage=0.995 --model_type=bpe


# step 3:

#CUDA_VISIBLE_DEVICES=1 python engines/rnnlm/train.py \
#	--train_file data/twitter/train_rnnlm_20k.txt \
#	--dev_file data/twitter/valid_rnnlm.txt \
#	--tokenizer_path data/twitter/sp_20k.model \
#	--output train.log \
#	--save_dir engines/rnnlm/models-20-rnn-200-step-30 \
#	--rnn_size 200 \
#	--num_layers 2 \
#	--model lstm \
#	--batch_size 20 \
#	--num_steps 30 \
#	--num_epochs 50 \
#	--validation_interval 1 \
#        --init_scale 0.1 \
#	--grad_clip 5.0 \
#	--learning_rate 1.0 \
#	--decay_rate 0.5 \
#	--keep_prob 0.5 \
#	--optimization sgd
#

CUDA_VISIBLE_DEVICES=1 python engines/rnnlm/train.py \
	--train_file data/twitter/train_clean_1M.txt \
	--dev_file data/twitter/valid_rnnlm.txt \
	--tokenizer_path data/twitter/sp_1m.model \
	--output train.log \
	--save_dir engines/rnnlm/models-1m-rnn-50-step-30 \
	--rnn_size 50 \
	--num_layers 2 \
	--model lstm \
	--batch_size 20 \
	--num_steps 30 \
	--num_epochs 50 \
	--validation_interval 1 \
        --init_scale 0.1 \
	--grad_clip 5.0 \
	--learning_rate 1.0 \
	--decay_rate 0.5 \
	--keep_prob 0.5 \
	--optimization sgd

#CUDA_VISIBLE_DEVICES=2 python engines/rnnlm/train.py \
#	--train_file data/twitter/train_rnnlm_100k.txt \
#	--dev_file data/twitter/valid_rnnlm.txt \
#	--tokenizer_path data/twitter/sp_100k.model \
#	--output train.log \
#	--save_dir engines/rnnlm/models-100 \
#	--rnn_size 200 \
#	--num_layers 2 \
#	--model lstm \
#	--batch_size 20 \
#	--num_steps 50 \
#	--num_epochs 50 \
#	--validation_interval 1 \
#        --init_scale 0.1 \
#	--grad_clip 5.0 \
#	--learning_rate 1.0 \
#	--decay_rate 0.2 \
#	--keep_prob 0.5 \
#	--optimization sgd



#python engines/bert/create_pretraining_data.py \
#   --input_file=data/twitter/valid_clean.txt \
#   --output_file=data/twitter/valid_clean_50.tfrecord \
#   --vocab_file=$BERT_BASE_DIR/vocab.txt \
#   --do_lower_case=True \
#   --max_seq_length=50 \
#   --max_predictions_per_seq=8 \
#   --masked_lm_prob=0.15 \
#   --random_seed=12345 \
#   --dupe_factor=5

#step 4:

#CUDA_VISIBLE_DEVICES=2 python engines/bert/run_pretraining.py \
#  --train_input_file=data/twitter/train_clean_50k_50.tfrecord \
#  --valid_input_file=data/twitter/valid_clean_50.tfrecord \
#  --output_dir=engines/bert/models/50k_50 \
#  --do_train=True \
#  --do_eval=True \
#  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
#  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
#  --train_batch_size=32 \
#  --max_seq_length=50 \
#  --max_predictions_per_seq=8 \
#  --num_train_steps=100000 \
#  --max_eval_steps=100000 \
#  --num_warmup_steps=1000 \
#  --learning_rate=2e-5
