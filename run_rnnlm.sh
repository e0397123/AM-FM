#!/bin/bash

# step 1:
#python engines/rnnlm/create_raw_data.py \
#  --train_file data/twitter/train.txt \
#  --valid_file data/twitter/valid.txt \
#  --train_output data/twitter/train_rnnlm_10k.txt \
#  --valid_output data/twitter/valid_rnnlm.txt \
#  --data_size 10000
#

# step 2:
spm_train --input=data/twitter/train_rnnlm_10k.txt --model_prefix=data/twitter/sp_10k --vocab_size=8000 --character_coverage=1.0 --model_type=bpe


# step 3:

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
