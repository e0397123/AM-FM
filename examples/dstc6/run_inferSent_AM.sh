#!/bin/bash


# general configuration
stage=2
stop_stage=100
version=2
CVD=0         # CUDA_VISIBLE_DEVICES

# config files


## pretrain related
tag="InferSent" # tag for managing experiments.
model_path="../../engines/embedding_models/InferSent"

# Data related
d_root=./Data

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

if [ ! -d ${model_path} ]; then
    echo "stage -1: Prepare InferSent"

    echo "stage -1-1: Clone InferSent"

    git clone https://github.com/facebookresearch/InferSent.git ${model_path}
    echo "stage -1-2: Download word embedding (default use fastText)"

#    mkdir GloVe
#    curl -Lo GloVe/glove.840B.300d.zip http://nlp.stanford.edu/data/glove.840B.300d.zip
#    unzip GloVe/glove.840B.300d.zip -d GloVe/
    mkdir ${model_path}/fastText
    curl -Lo ${model_path}/fastText/crawl-300d-2M.vec.zip https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
    unzip ${model_path}/fastText/crawl-300d-2M.vec.zip -d ${model_path}/fastText/

    echo "stage -1-3: Download pretrain model"
    mkdir ${model_path}/encoder
    curl -Lo ${model_path}/encoder/infersent1.pkl https://dl.fbaipublicfiles.com/infersent/infersent1.pkl
    curl -Lo ${model_path}/encoder/infersent2.pkl https://dl.fbaipublicfiles.com/infersent/infersent2.pkl

    echo "stage -1-4: Download get NLI data"
    bash ${model_path}/get_data.bash

fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Prepare modules"
    echo "stage 0-1: Install pytorch==1.2.0 nltk"
    pip install torch nltk

    echo "stage 0-1: Download nltk"
    python -c "import nltk;nltk.download('punkt')"

fi


# ================================================
# ref: http://www.gnu.org/software/bash/manual/html_node/Bash-Conditional-Expressions.html
# -s FILE:- FILE exists and has a size greater than zero
# -z string: True if the length of string is zero.
# -n string: True if the length of string is non-zero.
# -e file: True if file exists.
# -d file: True if file exists and is a directory.
# ================================================

evaluation_ref_file=${d_root}/clean_data/evaluation/ref_clean.txt
evaluation_hyp_file=${d_root}/clean_data/evaluation/hyp_clean.txt

pretrain_model_path=${model_path}/encoder/infersent${version}.pkl

if [ ${version} == 1 ]; then
    w2v_path=${model_path}/GloVe/glove.840B.300d.txt
elif [ ${version} == 2 ]; then
    w2v_path=${model_path}/fastText/crawl-300d-2M.vec
fi

result_file=./result/${tag}
mkdir -p ./result ${result_file}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Feature extraction.
    ### This step is to extract fixed word-level contextualized embedding.
    echo "stage 1: Feature extraction"
    CUDA_VISIBLE_DEVICES=${CVD} \
    python ${model_path}/extract_features.py \
        --version=${version}\
        --w2v_path=${w2v_path} \
        --model_path=${pretrain_model_path} \
        --out-dir=${result_file} \
        --tokenize \
        ${evaluation_ref_file} ${evaluation_hyp_file}

fi

evaluation_ref_json=${result_file}/$(basename ${evaluation_ref_file}).embs.npy
evaluation_hyp_json=${result_file}/$(basename ${evaluation_hyp_file}).embs.npy

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Compute AM Score"
    python ../../engines/embedding_models/InferSent/calc_am.py \
        --hyp_file="${evaluation_hyp_json}" \
        --ref_file="${evaluation_ref_json}"
fi
