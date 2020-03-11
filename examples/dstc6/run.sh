#!/bin/bash


# general configuration
stage=5
stop_stage=5
seed=12345    # random seed number
CVD=0         # CUDA_VISIBLE_DEVICES

# config files


## pretrain related
tag="bert" # tag for managing experiments.
pretrain_model_path="../../../model_pretrain/bert/uncased_L-12_H-768_A-12"

# Data related
d_root=./Data
data_size=10000
do_lower_case=True

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

#if [ ! -d ${pretrain_model_path} ]; then
#    echo "stage -1: pretrain model Download"
#    """download pretrain model here"""
#    echo "Code have not complete. Please set up the pretrain_model_path manually"
#    exit 1
#fi


#if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
#    echo "stage -0: Data Download"
#    mkdir -p ${d_root}
#    mkdir -p ${d_root}/raw_data
#    mkdir -p ${d_root}/clean_data
#    """download data here"""
#fi
#
#
#train_file_path=${d_root}/raw_data/train.txt
#valid_file_path=${d_root}/raw_data/valid.txt
#
#if [ ${data_size} == "all" ]; then
#    #TODO not complete
#    data_size=# length of data
#    train_output_path=${d_root}/clean_data/$(basename ${train_file_path%.*}_clean.txt)
#    valid_output_path=${d_root}/clean_data/$(basename ${valid_file_path%.*}_clean.txt)
#else
#    train_output_path=${d_root}/clean_data/$(basename ${train_file_path%.*}_clean${data_size}.txt)
#    valid_output_path=${d_root}/clean_data/$(basename ${valid_file_path%.*}_clean${data_size}.txt)
#fi
#
#if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
#    ### Create preprocessed training and validation data with specific training size.
#    ### This step is to conduct preprocessing on the twitter dialogues.
#    echo "stage 2: Data preparation"
#    python ../../engines/embedding_models/bert/create_raw_data.py \
#          --train_file=${train_file_path} \
#          --train_output=${train_output_path}\
#          --valid_file=${valid_file_path} \
#          --valid_output=${valid_output_path}\
#          --data_size=${data_size}
#fi
#
#train_file_path=${train_output_path}
#valid_file_path=${valid_output_path}


train_file_path=${d_root}/clean_data/train_clean_full.txt
valid_file_path=${d_root}/clean_data/valid_clean.txt

BERT_BASE_DIR=${pretrain_model_path}


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    ### Create tfrecord pretraining data.
    ### The tfrecord data is to easier the pretraining and faster loading.
    echo "stage 3: Create tfrecord pretraining data"
    python ../../engines/embedding_models/bert/create_pretraining_data.py \
        --input_file=${train_file_path} \
        --output_file=${train_file_path}.tfrecord \
        --vocab_file=${BERT_BASE_DIR}/vocab.txt \
        --do_lower_case=${do_lower_case} \
        --random_seed=${seed} \
        --max_seq_length=60 \
        --max_predictions_per_seq=9 \
        --masked_lm_prob=0.15 \
        --dupe_factor=5

    python ../../engines/embedding_models/bert/create_pretraining_data.py \
        --input_file=${valid_file_path} \
        --output_file=${valid_file_path}.tfrecord \
        --vocab_file=${BERT_BASE_DIR}/vocab.txt \
        --do_lower_case=${do_lower_case} \
        --random_seed=${seed} \
        --max_seq_length=60 \
        --max_predictions_per_seq=9 \
        --masked_lm_prob=0.15 \
        --dupe_factor=5
fi

train_file_path=${train_file_path}.tfrecord
valid_file_path=${valid_file_path}.tfrecord
pretrain_output_dir=./pretrain/${tag}

mkdir -p ./pretrain ${pretrain_output_dir}

echo "echo something"
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    ### Conduct pretraining of bert model
    echo "stage 4: Conduct pretraining of bert model"
    CUDA_VISIBLE_DEVICES=${CVD} \
    python ../../engines/embedding_models/bert/run_pretraining.py \
        --train_input_file=${train_file_path} \
        --valid_input_file=${valid_file_path} \
        --output_dir=${pretrain_output_dir} \
        --do_train=True \
        --do_eval=True \
        --bert_config_file=${BERT_BASE_DIR}/bert_config.json \
        --init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
        --train_batch_size=8 \
        --max_seq_length=60 \
        --max_predictions_per_seq=9 \
        --num_train_steps=5000 \
        --max_eval_steps=100 \
        --num_warmup_steps=100 \
        --learning_rate=2e-5
fi

# ================================================
# ref: http://www.gnu.org/software/bash/manual/html_node/Bash-Conditional-Expressions.html
# -s FILE:- FILE exists and has a size greater than zero
# -z string: True if the length of string is zero.
# -n string: True if the length of string is non-zero.
# -e file: True if file exists.
# -d file: True if file exists and is a directory.
# ================================================


#if [ -z ${tag} ]; then
#    expname=${train_set}_${backend}_$(basename ${train_config%.*})
#else
#    expname=${train_set}_${backend}_${tag}
#fi

evaluation_ref_file=${d_root}/clean_data/evaluation/ref_clean.txt
evaluation_hyp_file=${d_root}/clean_data/evaluation/hyp_clean.txt

bert_dstc_pretrain=${pretrain_output_dir}
bert_init_checkpoint=${bert_dstc_pretrain}

result_file=./result/${tag}
mkdir -p ./result ${result_file}

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    ### Feature extraction.
    ### This step is to extract fixed word-level contextualized embedding.
    echo "stage 5: Feature extraction"
    CUDA_VISIBLE_DEVICES=${CVD} \
    python ../../engines/embedding_models/bert/extract_features.py \
        --input_file=${evaluation_hyp_file} \
        --output_file=${result_file}/$(basename ${evaluation_hyp_file%.*})_feature.json \
        --vocab_file=${BERT_BASE_DIR}/vocab.txt \
        --bert_config_file=${BERT_BASE_DIR}/bert_config.json \
        --init_checkpoint=${bert_dstc_pretrain} \
        --layers=-1,-2,-3,-4 \
        --max_seq_length=60 \
        --batch_size=8 \
        --bert_model_dir=${bert_dstc_pretrain}

    CUDA_VISIBLE_DEVICES=${CVD} \
    python ../../engines/embedding_models/bert/extract_features.py \
        --input_file=${evaluation_ref_file} \
        --output_file=${result_file}/$(basename ${evaluation_ref_file%.*})_feature.json \
        --vocab_file=$BERT_BASE_DIR/vocab.txt \
        --bert_config_file=$BERT_BASE_DIR/bert_config.json \
        --init_checkpoint=${bert_dstc_pretrain} \
        --layers=-1,-2,-3,-4 \
        --max_seq_length=60 \
        --batch_size=8 \
        --estimator_output_dir=${result_file}
fi

evaluation_ref_json=${result_file}/$(basename ${evaluation_ref_file%.*})_feature.json
evaluation_hyp_json=${result_file}/$(basename ${evaluation_hyp_file%.*})_feature.json

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Compute AM Score"
    python ../../engines/embedding_models/bert/calc_am.py \
        --hyp_file=${evaluation_hyp_json} \
        --ref_file=${evaluation_ref_json} \
#        --strategy=
fi


# ===========================================
# FM part
# ===========================================

train_file_path=${d_root}/clean_data/train_clean_full.txt
valid_file_path=${d_root}/clean_data/valid_clean.txt
vocabulary_size=10000
mkdir -p ./pretrain/sentencepiece
spm_model_name=./pretrain/sentencepiece/twitter_${vocabulary_size}

if [ ${stage} -le 101 ] && [ ${stop_stage} -ge 101 ]; then
    # Follow https://github.com/google/sentencepiece.git
    # to train a sentencepiece tokenizer with the full training set
    echo "stage 1: Train a sentencepiece tokenizer"
    python ../../engines/preprocessing/spm_train.py \
          --input=${train_file_path}\
          --model_prefix=${spm_model_name} \
          --vocab_size=${vocabulary_size} \
          --character_coverage=0.995 \
          --model_type=bpe
fi

if [ ${stage} -le 102 ] && [ ${stop_stage} -ge 102 ]; then
    # Create preprocessed training and validation data
    # (same as step 1 in Using BERT Embedding Model)
    echo "stage 2: Create preprocessed data"
fi

model_name=./pretrain/rnnlm

hyp_out_file=./result/${model_name}/hyp_out
ref_out_file=./result/${model_name}/ref_out
embedding_name=embeddings_full.npy

if [ ${stage} -le 103 ] && [ ${stop_stage} -ge 103 ]; then
    # Create preprocessed training and validation data
    # (same as step 1 in Using BERT Embedding Model)
    echo "stage 3: Training the language model"
    SIZE=full
    CUDA_VISIBLE_DEVICES=${CVD} \
    python ../../engines/language_models/rnnlm/main.py \
        --data_path=${d_root}/clean_data \
        --data_size=${SIZE} \
        --model_path=${model_name}\
        --embedding_name=${embedding_name} \
        --tokenizer_path=${spm_model_name}.model \
        --bpe_vocab=${spm_model_name}.vocab \
        --hyp_out=${hyp_out_file} \
        --ref_out=${ref_out_file} \
        --batch_size=32 \
        --embedding_size=300 \
        --num_nodes=150,105,70 \
        --num_epochs=100 \
        --use_sp=True \
        --do_train=True \
        --do_eval=True \
        --do_dstc_eval=True

fi


if [ ${stage} -le 104 ] && [ ${stop_stage} -ge 104 ]; then
    echo "stage 6: Compute FM Score"
    python ../../engines/language_mode/calc_fm.py \
        --hyp_file=${hyp_out_file} \
        --ref_file=${ref_out_file}
fi