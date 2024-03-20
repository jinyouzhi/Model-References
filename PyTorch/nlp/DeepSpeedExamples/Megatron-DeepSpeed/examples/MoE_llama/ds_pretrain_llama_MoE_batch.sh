#!/bin/bash

set -ex

MY_SCRIPT_DIR="$( cd "$(dirname "$0")" && pwd )"

# Habana logs
#export HABANA_LOGS=${MY_SCRIPT_DIR}/habana_logs
#export LOG_LEVEL_ALL_HCL=0

# detect model recompile
#export PT_HPU_METRICS_FILE=${MY_SCRIPT_DIR}/hpu_metricslog.json
#export PT_HPU_METRICS_DUMP_TRIGGERS=process_exit,metric_change

# dump graph
#export GRAPH_VISUALIZATION=1

# 16x1b
llama_moe_16x1b() {
    MODEL_TO_TEST="1b"
    INPUT_TO_TEST="2048"
    GBS_TO_TEST="256"
    EXPERTS_TO_TEST="16"
    NODES_TO_RUN=1

    for m in ${MODEL_TO_TEST}; do
        for i in ${INPUT_TO_TEST}; do
            for g in ${GBS_TO_TEST}; do
                for e in ${EXPERTS_TO_TEST}; do
                    bash ${MY_SCRIPT_DIR}/ds_pretrain_llama_MoE.sh -m ${m} -s ${i} -n ${NODES_TO_RUN} \
                        -e ${e} \
                        --mbs 1 --gbs ${g} \
                        --fsdpa true --fsdpa-recompute false \
                        --fp8 false \
                        --zero 0 --ckpt 0 \
                        --eval-iter 0 --exit-interval 500 \
                        --load false --save false --log-interval 1 \
                        --optimizer fusedadamw --dropout 0 \
                        --use-hpu true
                done
            done
        done
    done
}

## 16x7b
llama_moe_16x7b() {
    #export PT_HPU_MAX_COMPOUND_OP_SIZE=10
    MODEL_TO_TEST="7b"
    INPUT_TO_TEST="2048"
    GBS_TO_TEST="256"
    EXPERTS_TO_TEST="16"
    NODES_TO_RUN=2
    LAYERS_TO_RUN=32

    for m in ${MODEL_TO_TEST}; do
        for i in ${INPUT_TO_TEST}; do
            for g in ${GBS_TO_TEST}; do
                for e in ${EXPERTS_TO_TEST}; do
                    bash ${MY_SCRIPT_DIR}/ds_pretrain_llama_MoE.sh -m ${m} -s ${i} -n ${NODES_TO_RUN} \
                        -l ${LAYERS_TO_RUN} -e ${e} \
                        --mbs 1 --gbs ${g} \
                        --fsdpa true --fsdpa-recompute false \
                        --fp8 false \
                        --zero 0 --ckpt 0 \
                        --eval-iter 0 --exit-interval 500 \
                        --load false --save false --log-interval 1 \
                        --optimizer fusedadamw --dropout 0 \
                        --use-hpu true
                done
            done
        done
    done
    #unset PT_HPU_MAX_COMPOUND_OP_SIZE
}

llama_moe_16x1b
#llama_moe_16x7b
