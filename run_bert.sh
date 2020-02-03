#!/bin/bash

## step 1:
#python engines/bert/create_raw_data.py \
#  --train_file data/twitter/train.txt \
#  --valid_file data/twitter/valid.txt \
# --train_output data/twitter/train_clean_500k.txt \
# --valid_output data/twitter/valid_clean.txt \
# --data_size 500000

#step 2:

#python engines/embedding_models/bert/create_pretraining_data.py \
#  --input_file=data/twitter/train_clean_50k.txt \
#  --output_file=data/twitter/train_clean_50k_70.tfrecord \
#  --vocab_file=$BERT_BASE_DIR/vocab.txt \
#  --do_lower_case=True \
#  --max_seq_length=70 \
#  --max_predictions_per_seq=11 \
#  --masked_lm_prob=0.15 \
#  --random_seed=12345 \
#  --dupe_factor=5
#
#step 3:

#python engines/embedding_models/bert/create_pretraining_data.py \
#   --input_file=data/twitter/valid_clean.txt \
#   --output_file=data/twitter/valid_clean_70.tfrecord \
#   --vocab_file=$BERT_BASE_DIR/vocab.txt \
#   --do_lower_case=True \
#   --max_seq_length=70 \
#   --max_predictions_per_seq=11 \
#   --masked_lm_prob=0.15 \
#   --random_seed=12345 \
#   --dupe_factor=5

#step 4:

CUDA_VISIBLE_DEVICES=2 python engines/embedding_models/bert/run_pretraining.py \
  --train_input_file=data/twitter/train_clean_50k_70.tfrecord \
  --valid_input_file=data/twitter/valid_clean_70.tfrecord \
  --output_dir=engines/bert/models/50k_70 \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=70 \
  --max_predictions_per_seq=11 \
  --num_train_steps=100000 \
  --max_eval_steps=100 \
  --num_warmup_steps=1000 \
  --learning_rate=2e-5
