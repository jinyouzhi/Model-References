#!/bin/bash

set -ex

MY_SCRIPT_DIR="$( cd "$(dirname "$0")" && pwd )"
MEGATRON_DEEPSPEED_DIR="$( cd ${MY_SCRIPT_DIR}/.. && pwd )"

PYTHON=python3
JSONL_PATH= #".../dataset/oscar-en-10k.jsonl"
LLAMA_TOKENIZER_MODEL= #".../dataset/llama2/tokenizer.model"
TARGET_DIR="${MEGATRON_DEEPSPEED_DIR}/data/oscar-en-10k-llama2"

mkdir -p "${TARGET_DIR}"

${PYTHON} ${MEGATRON_DEEPSPEED_DIR}/tools/preprocess_data.py --input ${JSONL_PATH} --output-prefix ${TARGET_DIR}/meg-llama2 --append-eod --tokenizer-type Llama2Tokenizer --tokenizer-model-file ${LLAMA_TOKENIZER_MODEL} --workers 64