#!/usr/bin/env sh

STORAGE=/home/nishida/projects/mllm-gesture-eval
STORAGE_DATA=${STORAGE}/data
STORAGE_RESULTS=${STORAGE}/results

DATASET=${STORAGE_DATA}/mscc/v1/dataset.json
# DATASET=${STORAGE_DATA}/mscc/v1/dataset_with_long_context_utterances.json

LLM_TYPE=openai
# LLM_TYPE=gemini
# LLM_TYPE=qwen
# LLM_TYPE=llava

MY_PREFIX=example

python step1_generate.py \
    --llm_type ${LLM_TYPE} \
    --dataset ${DATASET} \
    --results_dir ${STORAGE_RESULTS} \
    --prefix ${MY_PREFIX}

python step2_evaluate_by_llm.py \
    --input_file ${STORAGE_RESULTS}/${MY_PREFIX}/results.jsonl \
    --output_file ${STORAGE_RESULTS}/${MY_PREFIX}/evaluation_by_llm.jsonl

python step3_calc_evaluation_scores.py --input ${STORAGE_RESULTS}/${MY_PREFIX}/evaluation_by_llm.jsonl

