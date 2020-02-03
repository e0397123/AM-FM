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
2. create the data folder and add the raw train, validation and test file in the data folder.

### Run Adequacy Evaluation

#### Using BERT Embedding Model

1. Download the [BERT-Base, Multilingual Cased] pretrained model from https://github.com/google-research/bert and configure the BERT_BASE_DIR

2. create the preprocessed training and validation files with command: 
    
    python engines/embedding_models/bert/create_raw_data.py \
      --train_file data/twitter/train.txt \
      --train_output engines/embedding_models/bert/train_clean_100k.txt \
      --data_size 100000


