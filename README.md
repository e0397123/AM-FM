# AM-FM-PM

This repo is for the paper, 'Deep AM-FM: Toolkit For Automatic Dialogue Evaluation', IWSDS 2020 Submission and we are continuously improving this repo to make it a better platform for dialogue evaluation.

## The Deep AM-FM-PM Framework

### Adequacy Metric

This component aims to assess the semantic aspect of system responses, more specifically, how much source information is preserved by the dialogue generation with reference to human-written responses. The continuous space model is adopted for evaluating adequacy where good word-level or sentence-level embedding techniques are studied to measure the semantic closessness of system responses and human references in the continous vector space.

### Fluency Metric

This component aims to assess the syntactic validity of system responses. It tries to compare the system hypotheses against human references in terms of their respective sentence-level normalized log probabilities based on the assumption that sentences that are of similar syntactic validity should share similar perplexity level given by a language model. Hence, in this component, various language model techniques are explored to accurately estimate the sentence-level probability distribution.

### Pragmatics Metric

To be added


## Evaluation Procedure

### Toolkit Requirements

1. python 3.x
2. emoji=0.5.4
3. jsonlines=1.2.0
4. tensorflow-gpu=1.14.0
5. tqdm=4.38.0

### Dataset

1. Follow instructions at https://github.com/dialogtekgeek/DSTC6-End-to-End-Conversation-Modeling.git to collect the twitter dialogues.
2. Collect the evaluation dataset at https://www.dropbox.com/s/oh1trbos0tjzn7t/dstc6_t2_evaluation.tgz
2. Extract the training, validation and test dialogues into the data folder.

### Run Adequacy Evaluation

#### Using BERT Embedding Model (Most of the steps follow the google official bert repo)

1. Download the [BERT-Base, Multilingual Cased] pretrained model from https://github.com/google-research/bert and configure the BERT_BASE_DIR environment variable.

2. Create preprocessed training and validation data with specific training size: 
```bash
python engines/embedding_models/bert/create_raw_data.py \
  --train_file data/twitter/train.txt \
  --train_output data/twitter/train_clean_100k.txt \
  --data_size 100000
```

3. Create tfrecord pretraining data
```bash
python engines/embedding_models/bert/create_pretraining_data.py \
  --input_file=data/twitter/train_clean_100k.txt \
  --output_file=data/twitter/train_clean_100k_60.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=60 \
  --max_predictions_per_seq=9 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
```

4. Conduct pretraining
```bash
CUDA_VISIBLE_DEVICES=1 python engines/embedding_models/bert/run_pretraining.py \
  --train_input_file=data/twitter/train_clean_100k_60.tfrecord \
  --valid_input_file=data/twitter/valid_clean_60.tfrecord \
  --output_dir=engines/embedding_models/bert/models/50k_60_2 \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=8 \
  --max_seq_length=60 \
  --max_predictions_per_seq=9 \
  --num_train_steps=5000 \
  --max_eval_steps=100 \
  --num_warmup_steps=100 \
  --learning_rate=2e-5
```

5. Feature extraction
```bash
CUDA_VISIBLE_DEVICES=1 python engines/embedding_models/bert/extract_features.py \
  --input_file=data/twitter/dstc6_t2_evaluation/hypotheses/hyp_clean.txt \
  --output_file=engines/embedding_models/bert/features/hyp_clean_60_100k.jsonl \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=engines/embedding_models/bert/models/100k_60/model.ckpt-10000 \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=60 \
  --batch_size=8
```
```bash
CUDA_VISIBLE_DEVICES=1 python engines/embedding_models/bert/extract_features.py \
  --input_file=data/twitter/dstc6_t2_evaluation/references/ref_clean.txt \
  --output_file=engines/embedding_models/bert/features/ref_clean_60_100k.jsonl \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=engines/embedding_models/bert/models/100k_60/model.ckpt-10000 \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=60 \
  --batch_size=8
```

6. Compute AM Score
```bash
python engines/embedding_models/bert/calc_am.py \
  --hyp_file=engines/embedding_models/bert/features/hyp_clean_60_100k.jsonl \
  --ref_file=engines/embedding_models/bert/features/ref_clean_60_100k.jsonl \
  --strategy=top-layer-embedding-average
```

