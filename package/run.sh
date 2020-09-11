#!/bin/bash
# general configuration

#SBATCH --job-name=compute
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --output=compute.log


stage=1
stop_stage=7

# config files

# Data related
d_root=./data
LANG=gu
TASK=pmindia
SYSID=google
do_lower_case=True

## pretrain related
pretrain_model_path=${d_root}/multi_cased_L-12_H-768_A-12
BERT_BASE_DIR=${pretrain_model_path}
hyp_path=${d_root}/${TASK}/${LANG}/${SYSID}_hyp.txt
ref_path=${d_root}/${TASK}/${LANG}/ref.txt
num_test_cases=10


# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# ===========================================
# AM part
# ===========================================
am_model_path=./models/${TASK}/${LANG}/am
am_result_file=./result/${TASK}/${LANG}/am

if [ ! -d ${am_result_file} ]; then
	mkdir -p ${am_result_file}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Feature extraction.
    ### This step is to extract fixed word-level contextualized embedding.
    echo "stage 1: Feature extraction of hypothesis list"
	if [ ! -e ${am_result_file}/${SYSID}_hyp.jsonl ]; then
		/home/jiadong/anaconda3/envs/tensorflow-1.15/bin/python ./AM/extract_features.py \
        	--input_file ${hyp_path} \
        	--output_file ${am_result_file}/${SYSID}_hyp.jsonl \
        	--vocab_file ${BERT_BASE_DIR}/vocab.txt \
        	--bert_config_file ${BERT_BASE_DIR}/bert_config.json \
        	--init_checkpoint ${am_model_path}/model-best \
        	--layers -1 \
        	--max_seq_length 60 \
        	--batch_size 8 

	else
		echo "${am_result_file}/${SYSID}_hyp.jsonl already exists"
	fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Feature extraction of reference list"
	if [ ! -e ${am_result_file}/ref.jsonl ]; then
		/home/jiadong/anaconda3/envs/tensorflow-1.15/bin/python ./AM/extract_features.py \
         	--input_file ${ref_path} \
         	--output_file ${am_result_file}/ref.jsonl \
         	--vocab_file $BERT_BASE_DIR/vocab.txt \
         	--bert_config_file $BERT_BASE_DIR/bert_config.json \
         	--init_checkpoint ${am_model_path}/model-best \
         	--layers -1 \
         	--max_seq_length 60 \
         	--batch_size 8
	else
		echo "${am_result_file}/ref.jsonl already exists"
	fi
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Compute AM Score"
    python ./AM/calc_am.py \
        --hyp_file=${am_result_file}/${SYSID}_hyp.jsonl \
        --ref_file=${am_result_file}/ref.jsonl \
        --num_test=${num_test_cases} \
        --save_path=${am_result_file}/${SYSID}_am.score
fi

## # ===========================================
## # FM part
## # ===========================================

pretrained_xlm_path=${d_root}/xlmr.base
fm_model_path=./models/${TASK}/${LANG}/fm
fm_result_file=./result/${TASK}/${LANG}/fm

if [ ! -d ${fm_result_file} ]; then
	mkdir -p ${fm_result_file}
fi

## compute FM score
#
#

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
	echo "stage 4: Compute hypothesis sentence-level probability"
	/home/jiadong/anaconda3/envs/tensorflow-1.15/bin/python ./FM/eval.py \
		--data_path=${hyp_path} \
		--model_path=${fm_model_path} \
		--pretrained_model=${pretrained_xlm_path} \
		--tokenizer_path=${pretrained_xlm_path}/sentencepiece.bpe.model \
		--embedding_size=300 \
		--num_nodes=150,75 \
		--seq_len=50 \
		--save_path=${fm_result_file}/${SYSID}_hyp.prob

fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Compute reference sentence-level probability"
    /home/jiadong/anaconda3/envs/tensorflow-1.15/bin/python ./FM/eval.py \
        --data_path=${ref_path} \
        --model_path=${fm_model_path} \
        --pretrained_model=${pretrained_xlm_path} \
        --tokenizer_path=${pretrained_xlm_path}/sentencepiece.bpe.model \
        --embedding_size=300 \
        --num_nodes=150,75 \
		--seq_len=50 \
        --save_path=${fm_result_file}/ref.prob
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Compute FM Score"
    python ./FM/calc_fm.py \
        --hyp_file=${fm_result_file}/${SYSID}_hyp.prob \
        --ref_file=${fm_result_file}/ref.prob \
        --num_test=${num_test_cases} \
        --save_path=${fm_result_file}/${SYSID}_fm.score
fi


# ===========================================
# combined both scores
# ===========================================

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then

    echo "stage 7: Combine AM & FM scores"
    python amfm.py \
        --am_score=${am_result_file}/${SYSID}_am.score \
        --fm_score=${fm_result_file}/${SYSID}_fm.score \
        --lambda_value=0.5 \
		--save_path=./result/${TASK}/${LANG}/${SYSID}_amfm.score
fi

echo "Thank you for using Deep AMFM Framework"
