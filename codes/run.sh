#!/usr/bin/env sh

EXAMPLES=/home/nishida/projects/mllm-gesture-eval/data/mscc/v1/examples.json
# EXAMPLES=/home/nishida/projects/mllm-gesture-eval/data/mscc/v1/examples_with_long_context_utterances.json

RESULTS_DIR=/home/nishida/projects/mllm-gesture-eval/results

LLM_TYPE=openai
# LLM_TYPE=gemini
# LLM_TYPE=qwen
# LLM_TYPE=llava

MY_PREFIX=example
# MY_PREFIX=Jan09_08-42-25 # GPT-4o
# MY_PREFIX=Jan09_17-40-18 # GPT-4o + long context utterances
# MY_PREFIX=Jan09_19-58-26 # GPT-4o + physical perspective description
# MY_PREFIX=Jan10_03-08-03 # GPT-4o + gesture type
# MY_PREFIX=Jan08_19-21-54 # GPT-4o-mini
# MY_PREFIX=Jan09_12-46-44 # Gemini-1.5-pro
# MY_PREFIX=Jan08_23-06-53 # Gemini-1.5-flash
# MY_PREFIX=gpt4omini_v1
# MY_PREFIX=gpt4o_v1
# MY_PREFIX=gemini15flash_v1
# MY_PREFIX=gemini15pro_v1
# MY_PREFIX=gpt4o_long_context_v1
# MY_PREFIX=gpt4o_physical_desc_v1
# MY_PREFIX=gpt4o_gesture_type_v1
# MY_PREFIX=gpt4o_without_dialogue_v1
# MY_PREFIX=gpt4o_without_vision_v1
# MY_PREFIX=qwen_v1
# MY_PREFIX=llava_v1
# MY_PREFIX=gpt4o_english_v1

python step1_generate.py \
    --llm_type ${LLM_TYPE} \
    --examples ${EXAMPLES} \
    --results_dir ${RESULTS_DIR} \
    --prefix ${MY_PREFIX}

python step2_evaluate_by_llm.py \
    --input_file ${RESULTS_DIR}/${MY_PREFIX}/results.jsonl \
    --output_file ${RESULTS_DIR}/${MY_PREFIX}/evaluation_by_llm.jsonl

python step3_calc_evaluation_scores.py --input ${RESULTS_DIR}/${MY_PREFIX}/evaluation_by_llm.jsonl

