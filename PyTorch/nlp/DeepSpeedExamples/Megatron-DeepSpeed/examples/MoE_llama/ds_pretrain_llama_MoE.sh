#!/bin/bash
set -x

DIR="$( cd "$(dirname "$0")" && pwd )"

MEGATRON_DEEPSPEED_DIR="$( cd ${DIR}/../.. && pwd )"

# do not show me git error
#git config --global --add safe.directory '*'

# Test setting
DEFAULT_MODEL=1b
DEFAULT_LAYERS=
DEFAULT_SEQ_LEN=2048
DEFAULT_NODES=1
DEFAULT_TP=1
DEFAULT_PP=1
DEFAULT_DP=1
DEFAULT_SP=0
DEFAULT_EP=
DEFAULT_USE_FUSED_SDPA=false
DEFAULT_USE_FUSED_SDPA_RECOMPUTE=false
DEFAULT_MBS=4
DEFAULT_GBS=256
DEFAULT_ZERO=0
DEFAULT_HPU_FP8_TRANSFORMER_ENGINE=false
DEFAULT_CKPT_ACT=0
DEFAULT_TRAIN_ITER=500000
DEFAULT_EVAL_ITER=100
DEFAULT_EXIT_INTERVAL=0
DEFAULT_LOAD=true
DEFAULT_SAVE=true
DEFAULT_LOG_INTERVAL=10
DEFAULT_EXPERTS=16
DEFAULT_EXPERT_INTERVAL=1
DEFAULT_USE_HPU=true
DEFAULT_HOSTSFILE="${DIR}/hostsfile"
DEFAULT_OPTIMIZER="adamw"
DEFAULT_DROPOUT=0.1
DEFAULT_TOPK_GATE=1

TEST_MODEL=${DEFAULT_MODEL}
LAYERS=${DEFAULT_LAYERS}
SEQ_LEN=${DEFAULT_SEQ_LEN}
NODES=${DEFAULT_NODES}
TP=${DEFAULT_TP}
PP=${DEFAULT_PP}
DP=${DEFAULT_DP}
SP=${DEFAULT_SP}
EP=${DEFAULT_EP}
USE_FUSED_SDPA=${DEFAULT_USE_FUSED_SDPA}
USE_FUSED_SDPA_RECOMPUTE=${DEFAULT_USE_FUSED_SDPA_RECOMPUTE}
MBS=${DEFAULT_MBS}
GBS=${DEFAULT_GBS}
ZERO=${DEFAULT_ZERO}
HPU_FP8_TRANSFORMER_ENGINE=${DEFAULT_HPU_FP8_TRANSFORMER_ENGINE}
CKPT_ACT=${DEFAULT_CKPT_ACT}
TRAIN_ITER=${DEFAULT_TRAIN_ITER}
EVAL_ITER=${DEFAULT_EVAL_ITER}
EXIT_INTERVAL=${DEFAULT_EXIT_INTERVAL}
LOAD=${DEFAULT_LOAD}
SAVE=${DEFAULT_SAVE}
LOG_INTERVAL=${DEFAULT_LOG_INTERVAL}
EXPERTS=${DEFAULT_EXPERTS}
EXPERT_INTERVAL=${DEFAULT_EXPERT_INTERVAL}
USE_HPU=${DEFAULT_USE_HPU}
HOSTSFILE=${DEFAULT_HOSTSFILE}
OPTIMIZER=${DEFAULT_OPTIMIZER}
DROPOUT=${DEFAULT_DROPOUT}
TOPK_GATE=${DEFAULT_TOPK_GATE}
UNIV_CP=0
USE_TORCH_COMPILE=false

# Profile
PROFILE_ARGS=
# pt, pt-full, hltv
#PROFILE_ARGS="--profile pt-full --profile-steps 2,3"

show_help() {
    echo "Usage: $0 [OPTIONS]
This script is used to run llama moe.

Options:
-m, --model <model>                 Model: 1b(16x1b, default),7b(16x7b).
-l, --layer <layer>                 Custom layer numbers.
-s,--seq-len <len>                  Seq len.
-n,--node <node>                    Nodes.
-e,--experts <ep>                   Number of experts.
--expert-interval <interval>        Use experts in every \"expert-interval\" layers, default 1.
--tp <tp>                           TP.
--pp <pp>                           PP.
--dp <dp>                           DP.
--sp <sp>                           Seq parallel.
--ep <ep>                           Expert parallel.
--fsdpa <true|false>                FusedSDPA.
--fsdpa-recompute <true|false>      FusedSDPA recompute.
--mbs <mbs>                         Micro batch size.
--gbs <gbs>                         Global batch size.
--zero <zero>                       DeepSpeed ZeRO.
--fp8 <true|false>                  FP8 training.
--ckpt <ckpt>                       Activation checkpoint.
--train-iter                        Training iterations.
--eval-iter                         Eval iterations.
--exit-interval                     Exit after interval iterations.
--load <true|false>                 Load checkpoint.
--save <true|false>                 Save checkpoint.
--log-interval <log>                Log interval.
--use-hpu <true|false>              Use Habana HPU.
--hostsfile <path>                  Path for hostsfile.
--optimizer <opti>                  Optimizer, default adamw.
--dropout <dropout>                 Dropout for attention & hidden, default 0.1.
--topk <topk>                       Sets the k in TopK gating for MoE layers, default 1.
-h, --help                          Show this help text.
"
}

die(){
    local _EXIT_CODE=$(( $? == 0 ? 99 : $? ))
    stderr "ERROR: $*"
    exit ${_EXIT_CODE}
}

while [ $# -gt 0 ] ; do
    case "$1" in
        -m|--model)
            TEST_MODEL="${2:-${DEFAULT_MODEL}}"
            shift; shift;;
        -l|--layer)
            LAYERS="${2:-${DEFAULT_LAYERS}}"
            shift; shift;;
        -s|--seq-len)
            SEQ_LEN="${2:-${DEFAULT_SEQ_LEN}}"
            shift; shift;;
        -n|--node)
            NODES="${2:-${DEFAULT_NODES}}"
            shift; shift;;
        -e|--experts)
            EXPERTS="${2:-${DEFAULT_EXPERTS}}"
            shift; shift;;
        -e|--expert-interval)
            EXPERT_INTERVAL="${2:-${DEFAULT_EXPERT_INTERVAL}}"
            shift; shift;;
        --tp)
            TP="${2:-${DEFAULT_TP}}"
            shift; shift;;
        --pp)
            PP="${2:-${DEFAULT_PP}}"
            shift; shift;;
        --dp)
            DP="${2:-${DEFAULT_DP}}"
            shift; shift;;
        --sp)
            SP="${2:-${DEFAULT_SP}}"
            shift; shift;;
        --ep)
            EP="${2:-${DEFAULT_EP}}"
            shift; shift;;
        --fsdpa)
            USE_FUSED_SDPA="${2:-${DEFAULT_USE_FUSED_SDPA}}"
            shift; shift;;
        --fsdpa-recompute)
            USE_FUSED_SDPA_RECOMPUTE="${2:-${DEFAULT_USE_FUSED_SDPA_RECOMPUTE}}"
            shift; shift;;
        --mbs)
            MBS="${2:-${DEFAULT_MBS}}"
            shift; shift;;
        --gbs)
            GBS="${2:-${DEFAULT_GBS}}"
            shift; shift;;
        --zero)
            ZERO="${2:-${DEFAULT_ZERO}}"
            shift; shift;;
        --fp8)
            HPU_FP8_TRANSFORMER_ENGINE="${2:-${DEFAULT_HPU_FP8_TRANSFORMER_ENGINE}}"
            shift; shift;;
        --ckpt)
            CKPT_ACT="${2:-${DEFAULT_CKPT_ACT}}"
            shift; shift;;
        --train-iter)
            TRAIN_ITER="${2:-${DEFAULT_TRAIN_ITER}}"
            shift; shift;;
        --eval-iter)
            EVAL_ITER="${2:-${DEFAULT_EVAL_ITER}}"
            shift; shift;;
        --exit-interval)
            EXIT_INTERVAL="${2:-${DEFAULT_EXIT_INTERVAL}}"
            shift; shift;;
        --load)
            LOAD="${2:-${DEFAULT_LOAD}}"
            shift; shift;;
        --save)
            SAVE="${2:-${DEFAULT_SAVE}}"
            shift; shift;;
        --log-interval)
            LOG_INTERVAL="${2:-${DEFAULT_LOG_INTERVAL}}"
            shift; shift;;
        --use-hpu)
            USE_HPU="${2:-${DEFAULT_USE_HPU}}"
            shift; shift;;
        --hostsfile)
            HOSTSFILE="${2:-${DEFAULT_HOSTSFILE}}"
            shift; shift;;
        --optimizer)
            OPTIMIZER="${2:-${DEFAULT_OPTIMIZER}}"
            shift; shift;;
        --dropout)
            DROPOUT="${2:-${DEFAULT_DROPOUT}}"
            shift; shift;;
        --topk)
            TOPK_GATE="${2:-${DEFAULT_TOPK_GATE}}"
            shift; shift;;
        -h|--help)
            show_help
            exit 0;;
        *)
            show_help && die "Unknown parameter '$1'"
    esac
done
###############################################################################
### Main configs
## LLaMA Model Configs
if [ "${TEST_MODEL}" == "1b" ]; then
NUM_LAYERS=4
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=11008
NUM_ATTN_HEADS=32
NUM_KVHEADS=32
LR=3e-4
MIN_LR=3e-5
#INIT_STD=0.014

elif [ "${TEST_MODEL}" == "7b" ]; then
NUM_LAYERS=32
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=11008
NUM_ATTN_HEADS=32
NUM_KVHEADS=32
LR=3e-4
MIN_LR=3e-5
# INIT_STD=0.01
fi

## override layers
if [ -n "${LAYERS}" ]; then
    NUM_LAYERS=${LAYERS}
fi

###############################################################################
### Training duration configs
## Keep using default iteration based settings, use exit-interval to stop early.
## Another termination condition in minutes. Set it large enough to avoid
## undesired early termination.
EXIT_DURATION=30000000
###############################################################################
### LR configs
LR_WARMUP_ITERS=2000
###############################################################################
### Parallelism configs
## Micro batch size per GPU
## Make sure that BATCH_SIZE <= GBS*PP_SIZE*MP_SIZE/NUM_GPUS
BATCH_SIZE=${MBS}

## Model parallelism, 1 is no MP
## Currently MoE models have divergence issue when MP > 1.
MP_SIZE=${TP}

## Pipeline parallelism
## Currently we don't support PP for MoE. To disable PP, set PP_SIZE
## to 1 and use the "--no-pipeline-parallel" arg.
PP_SIZE=${PP}
NGPU_PER_NODE=8
NUM_GPUS=$(( ${NODES} * ${NGPU_PER_NODE} ))
###############################################################################
### MoE configs
## Number of experts. EP_SIZE 1 means dense model without MoE
EP_SIZE=${EXPERTS}

if [[ $EP_SIZE -gt $NUM_GPUS ]]; then
    EP_PARALLEL_SIZE=$NUM_GPUS
else
    EP_PARALLEL_SIZE=$EP_SIZE
fi

if [ -n "${EP}" ]; then
    EP_PARALLEL_SIZE=${EP}
fi

## Coefficient for MoE loss.
MLC=0.01

## Below configs adjust the MoE expert token capacity limit during training and
## eval. To completely disable capacity limit, set MOE_DROP_TOKEN to false.
## Larger capacity factor or disabling capacity limit could improve training
## convergence, but will also reduce training throughput.
MOE_TRAIN_CAP_FACTOR=1.0
MOE_EVAL_CAP_FACTOR=1.0
MOE_MIN_CAP=4
MOE_DROP_TOKEN="true"
# MOE_DROP_TOKEN="false"

### Misc configs
#LOG_INTERVAL=10
LOG_INTERVAL=${LOG_INTERVAL}
#EVAL_ITERS=10
EVAL_ITERS=${EVAL_ITER}
EVAL_INTERVAL=1000
SAVE_INTERVAL=2000

## Activation checkpointing saves GPU memory, but reduces training speed
ACTIVATION_CHECKPOINT="false"
if [[ ${CKPT_ACT} -eq 1 || ${CKPT_ACT} -eq 2 ]]; then
    ACTIVATION_CHECKPOINT="true"
fi
###############################################################################
### Output and data configs
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
host="${HOSTNAME}"
NAME="llama-${TEST_MODEL}-lr-${LR}-minlr-${MIN_LR}-gbs-${GBS}-gpus-${NUM_GPUS}-mp-${MP_SIZE}-pp-${PP_SIZE}"
if [[ $EP_SIZE -gt 1 ]]; then
    NAME="${NAME}-ep-${EP_SIZE}-mlc-${MLC}-cap-${MOE_TRAIN_CAP_FACTOR}-drop-${MOE_DROP_TOKEN}"
fi

OUTPUT_BASEPATH=$DIR/output
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}_${host}_${current_time}"
mkdir -p ${TENSORBOARD_DIR}
## Note that for MoE model with billion-scale base model, the checkpoint can be
## as large as TB-scale which normal NFS cannot handle efficiently.
CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"

### Dataset
VOCAB_PATH=${DIR}/dataset/oscar-en-10k/gpt2-vocab.json
MERGE_PATH=${DIR}/dataset/oscar-en-10k/gpt2-merges.txt
DATA_BLEND=${DIR}/dataset/oscar-en-10k/meg-gpt2_text_document
###############################################################################
data_options=" \
         --vocab-file ${VOCAB_PATH} \
         --merge-file ${MERGE_PATH} \
         --data-path ${DATA_BLEND} \
         --data-impl mmap"

if [ "$USE_HPU" == "true" ]; then
    HPU_ARGS="--use_hpu --distributed-backend=hccl --hpu-deterministic \
        --no-masked-softmax-fusion --no-bias-gelu-fusion --no-bias-dropout-fusion"
fi

#TRAINING_PRECISION="--fp16"
TRAINING_PRECISION="--bf16"
CONFIG_FP16_ENABLED=false
CONFIG_BF16_ENABLED=true
LOAD_CHECKPOINT=${LOAD}
LOAD_CHECKPOINT_ARG=
if [ "${LOAD_CHECKPOINT}" == "true" ]; then
    LOAD_CHECKPOINT_ARG="--load ${CHECKPOINT_PATH}"
fi
SAVE_CHECKPOINT=${SAVE}
SAVE_CHECKPOINT_ARG=
if [ "${SAVE_CHECKPOINT}" == "true" ]; then
    SAVE_CHECKPOINT_ARG="--save ${CHECKPOINT_PATH} \
        --save-interval ${SAVE_INTERVAL} \
        --verify-checkpoint --verify-checkpoint-model-type LLAMA"
fi

# fp8
USE_HPU_FP8_TRANSFORMER_ENGINE_ARG=
if [[ ${HPU_FP8_TRANSFORMER_ENGINE} == "true" ]]; then
    USE_HPU_FP8_TRANSFORMER_ENGINE_ARG="--use-hpu-fp8-transformer-engine"
fi

megatron_options=" \
        --no-bias \
        --layernorm-type rmsnorm \
        --activation-func-type swiglu \
        --layernorm-epsilon 1e-6 \
        --optimizer ${OPTIMIZER} \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --adam-eps 1e-5 \
        --tensor-model-parallel-size ${MP_SIZE} \
        --moe-expert-parallel-size ${EP_PARALLEL_SIZE} \
        --num-experts ${EP_SIZE} \
        --expert-interval ${EXPERT_INTERVAL} \
        --moe-loss-coeff ${MLC} \
        --topk ${TOPK_GATE} \
        --position-embedding-type rotary \
        --moe-train-capacity-factor ${MOE_TRAIN_CAP_FACTOR} \
        --moe-eval-capacity-factor ${MOE_EVAL_CAP_FACTOR} \
        --moe-min-capacity ${MOE_MIN_CAP} \
        --micro-batch-size ${BATCH_SIZE} \
        --exit-duration-in-mins ${EXIT_DURATION} \
        --global-batch-size ${GBS} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --num-key-value-heads ${NUM_KVHEADS} \
        --seq-length ${SEQ_LEN} \
        --train-iters ${TRAIN_ITER} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --lr-warmup-iters ${LR_WARMUP_ITERS} \
        --log-interval ${LOG_INTERVAL} \
        --eval-interval ${EVAL_INTERVAL} \
        --eval-iters ${EVAL_ITERS} \
        --exit-interval $EXIT_INTERVAL \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --attention-dropout ${DROPOUT} \
        --hidden-dropout ${DROPOUT} \
        --no-query-key-layer-scaling \
        ${TRAINING_PRECISION} \
        ${LOAD_CHECKPOINT_ARG} \
        ${SAVE_CHECKPOINT_ARG} \
        --use-torch-compile $USE_TORCH_COMPILE \
        --use-fused-sdpa ${USE_FUSED_SDPA} \
        --use-fused-sdpa-with-recompute ${USE_FUSED_SDPA_RECOMPUTE} \
        ${USE_HPU_FP8_TRANSFORMER_ENGINE_ARG} \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensorboard-dir ${TENSORBOARD_DIR}"

if [ ${SP} -eq 1 ]
then
megatron_options="${megatron_options} \
        --sequence-parallel"
fi

if [ $UNIV_CP -eq 1 ]
then
echo "Loading Universal Checkpoint from ${CHECKPOINT_PATH}"
megatron_options="${megatron_options} \
        --universal-checkpoint"
fi

if [ "${ACTIVATION_CHECKPOINT}" = "true" ]; then
megatron_options="${megatron_options} \
        --checkpoint-activations"
fi

if [[ $EP_SIZE -gt 1 ]]; then
megatron_options="${megatron_options} \
        --create-moe-param-group"
fi

if [ "${MOE_DROP_TOKEN}" = "false" ]; then
megatron_options="${megatron_options} \
        --disable-moe-token-dropping"
fi

PARTITIONED_MODE="true"
if [ ${SP} -eq 1 ]; then
    PARTITIONED_MODE="false"
fi

template_json="ds_config_llama_TEMPLATE.json"
config_json="${OUTPUT_BASEPATH}/ds_config_llama_${NAME}.json"
sed "s/CONFIG_BATCH_SIZE/${GBS}/" ${template_json} \
    | sed "s/CONFIG_MBSIZE/${BATCH_SIZE}/" \
    | sed "s/LOG_INTERVAL/${LOG_INTERVAL}/" \
    | sed "s/ZERO_STAGE/${ZERO}/" \
    | sed "s/CONFIG_FP16_ENABLED/${CONFIG_FP16_ENABLED}/" \
    | sed "s/CONFIG_BF16_ENABLED/${CONFIG_BF16_ENABLED}/" \
    | sed "s/CONFIG_PARTITIONED_MODE/${PARTITIONED_MODE}/" \
    > ${config_json}

deepspeed_options=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage=${ZERO} \
    --pipeline-model-parallel-size ${PP_SIZE}"

# Currently MoE is not compatible with pipeline parallel
if [[ $EP_SIZE -gt 1 ]]; then
deepspeed_options="${deepspeed_options} \
        --no-pipeline-parallel"
fi

if [ "${ACTIVATION_CHECKPOINT}" = "true" ]; then
    if [ $CKPT_ACT -eq 1 ]; then
        deepspeed_options="${deepspeed_options} \
                --deepspeed-activation-checkpointing"
    elif [ $CKPT_ACT -eq 2 ]; then
        deepspeed_options="${deepspeed_options} \
                --deepspeed-activation-checkpointing --checkpoint-activations-granularity selective"
    fi
fi

# multi nodes
MULTINODE_CMD=""
if [ "${NODES}" -ne "1" -a -f "${HOSTSFILE}" ]; then
    MULTINODE_CMD="--num_nodes ${NODES} \
                   --hostfile=${HOSTSFILE} \
                   --master_addr $(head -n 1 ${HOSTSFILE} | sed -n s/[[:space:]]slots.*//p) --master_port 29500"
fi

if [ -n "${MULTINODE_CMD}" ]; then
    if [ -f ${MEGATRON_DEEPSPEED_DIR}/sync_workspace.sh ]; then
        ${MEGATRON_DEEPSPEED_DIR}/sync_workspace.sh
    fi
fi

CMD="python3 -u ${MEGATRON_DEEPSPEED_DIR}/pretrain_llama.py ${megatron_options} ${data_options} ${deepspeed_options} ${HPU_ARGS} ${PROFILE_ARGS}"

# run!
deepspeed ${MULTINODE_CMD} \
          --num_gpus ${NGPU_PER_NODE} \
          --no_local_rank \
          --no_python \
          /usr/bin/bash -c "$CMD" 2>&1 | tee ${OUTPUT_BASEPATH}/log/${NAME}_${host}_${current_time}.log
