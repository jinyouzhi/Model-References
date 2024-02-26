# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company.

#!/bin/bash


set -ex

# -------------------------------------------
# LLaMA 2 70B training script
# Paper: https://arxiv.org/pdf/2307.09288.pdf
# -------------------------------------------

# ----------------------
# Configurable parameters
DATA_DIR=${HL_DATA_DIR_ROOT:-/data/bigscience/oscar-en/}
DATA_IDX_DIR=${HL_DATA_IDX_DIR:-}
DATA_FILE_PREFIX=${HL_DATA_FILE_PREFIX:-meg-gpt2_text_document}
NUM_NODES=${HL_NUM_NODES:-1}
DP=${HL_DP:-1}
TP=${HL_TP:-8}
PP=${HL_PP:-1}
MICRO_BATCH=${HL_MICRO_BATCH:-1}
GLOBAL_BATCH=${HL_GBS:-1024}
SEQ_LEN=${HL_SEQ_LEN:-4096}
EXIT_INTERVAL=${HL_EXIT_INTERVAL:-0}
OUTPUT_DIR=${HL_RESULTS_DIR:-}
OUTPUT_DIR_PREFIX=${HL_RESULTS_DIR_PREFIX:-.}
CHECKPOINT_SAVE=${HL_SAVE:-1}
SAVE_INTERVAL=${HL_SAVE_INTERVAL:-2000}
CHECKPOINTS_DIR=${HL_CHECKPOINTS_DIR:-}
TENSORBOARD_DIR=${HL_TENSORBOARD_DIR:-}
KILL_SWITCH_FILE=${HL_KILL_SWITCH:-}
HOSTSFILE=${HL_HOSTSFILE:-}
USE_HPU=${HL_USE_HPU:-1}
CKP_ACT=${HL_CKP_ACT:-0}
UNIV_CP=${HL_UNIV_CP:-0}
QNPU_DIR=${HL_QNPU_DIR:-}
LOG_INTERVAL=${HL_LOG_INTERVAL:-10}
LLAMA_VER=${HL_LLAMA_VER:-70}
N_LAYERS=${HL_NUM_LAYERS:-80}
N_GPU_PER_NODE=${HL_NGPU_PER_NODE:-8}
ZERO_STAGE=${HL_ZERO_STAGE:-0}
POS_EMB_TYPE=${HL_POSITION_EMBEDDING_TYPE:-rotary}
PROFILE=${HL_PROFILE:-} #provide either of pt, pt-full, hltv such as HL_PROFILE=hltv
SEQ_PARALLEL=${HL_SEQ_PARALLEL:-0} #set to 1 to enable sequence parallelism
OPTIMIZER=${HL_OPTIMIZER:-adamw}
USE_FUSED_SDPA=${HL_USE_FUSED_SDPA:-true}
USE_FUSED_SDPA_WITH_RECOMPUTE=${HL_USE_FUSED_SDPA_WITH_RECOMPUTE:-false}
DROPOUT=${HL_DROPOUT:-0.1}
USE_TORCH_COMPILE=${HL_USE_TORCH_COMPILE:-false}
NO_PIPELINE_PARALLEL=${HL_NO_PIPELINE_PARALLEL:-0}
DATA_TOKENIZER_TYPE=${HL_DATA_TOKENIZER_TYPE:-} # GPT2BPETokenizer(default) or Llama2Tokenizer.
# ----------------------

if [[ -z "$MODEL_REFERENCES_ROOT" ]]; then
    echo "Must provide MODEL_REFERENCES_ROOT in environment" 1>&2
    exit 1
fi

DATA_PATH=${DATA_DIR}/${DATA_FILE_PREFIX}
MODEL_DIR=$MODEL_REFERENCES_ROOT/PyTorch/nlp/DeepSpeedExamples/Megatron-DeepSpeed

# Scaling
NGPU_PER_NODE=${N_GPU_PER_NODE}
NUM_GPUs=$(($DP * $TP * $PP))

if [ $LLAMA_VER -eq 70 ]; then
    # Llama2-70B model architecture
    NLAYERS=${N_LAYERS} # must be divisible by PP; set to 80 for 70B
    NHIDDEN=8192
    NHEADS=64 # must be divisible by TP
    N_KVHEADS=8 # must be divisible by TP
    FFN_HIDDEN_SIZE=28672
else
    # Llama2-13B model architecture
    NLAYERS=${N_LAYERS} # must be divisible by PP; set to 40 for 13B
    NHIDDEN=5120
    NHEADS=40 # must be divisible by TP
    N_KVHEADS=40 # must be divisible by TP
    FFN_HIDDEN_SIZE=13824
fi

# Experiment name
if [ -z "$EXP_NAME" ]; then
    EXP_NAME="default"
fi

# output paths
if [ -z "$OUTPUT_DIR" ]; then
    RUNTIME=`date +"%Y%m%d_%H%M"`
    OUTPUT_DIR=${OUTPUT_DIR_PREFIX}/out/llamav2_${LLAMA_VER}b/ds_${EXP_NAME}_z${ZERO_STAGE}_nl${NLAYERS}_hs${HIDDEN}_gb${GLOBAL_BATCH}_mb${MICRO_BATCH}_sp${SEQ_PARALLEL}_D${DP}_T${TP}_P${PP}_GPUs${NUM_GPUs}_${RUNTIME}
fi

if [ -z "$CHECKPOINTS_DIR" ]; then
    CHECKPOINTS_DIR=$OUTPUT_DIR/checkpoints
fi

if [ -z "$TENSORBOARD_DIR" ]; then
    TENSORBOARD_DIR=$OUTPUT_DIR/tensorboard
fi

if [[ "$POS_EMB_TYPE" == "alibi" ]]; then
    MAX_POS_EMBEDDING="--max-position-embeddings $SEQ_LEN"
    POS_EMB_TYPE="alibi $MAX_POS_EMBEDDING"
fi

mkdir -p ${OUTPUT_DIR}
mkdir -p ${TENSORBOARD_DIR}

# handle kill switch argument
if [ -z "$KILL_SWITCH_FILE"]; then
    KILL_SWITCH_ARG=""
else
    KILL_SWITCH_ARG="--kill-switch-path $KILL_SWITCH_FILE"
fi

PARTITIONED_MODE="true"
if [ $SEQ_PARALLEL -eq 1 ]; then
    PARTITIONED_MODE="false"
fi

# create DS config
DS_CONFIG=${OUTPUT_DIR}/ds_config.json
cat << EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": $LOG_INTERVAL,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "bf16": {
    "enabled": true,
    "accumulate_grads_via_hooks": true
  },
  "fp16": {"enabled": false},
  "wall_clock_breakdown": false,
  "pipeline": {
    "pipe_partitioned": $PARTITIONED_MODE,
    "grad_partitioned": $PARTITIONED_MODE
  }
}
EOT

# configure multi-node
MULTINODE_CMD=""
if [ "$NUM_NODES" -ne "1" -a -f "$HOSTSFILE" ]; then
    MULTINODE_CMD="--hostfile=$HOSTSFILE \
                   --master_addr $(head -n 1 $HOSTSFILE | sed -n s/[[:space:]]slots.*//p) "
fi

# dataset
DATA_ARGS="--data-path ${DATA_PATH}"
if [ -z "${DATA_TOKENIZER_TYPE}" ]; then
    DATA_ARGS="${DATA_ARGS} --vocab-file $DATA_DIR/gpt2-vocab.json --merge-file $DATA_DIR/gpt2-merges.txt"
elif [ "${DATA_TOKENIZER_TYPE}" == "Llama2Tokenizer" ]; then
    DATA_ARGS="${DATA_ARGS} --tokenizer-type ${DATA_TOKENIZER_TYPE} --tokenizer-model-file ${DATA_DIR}/tokenizer.model"
else
    echo "Unsupported tokenizer type ${DATA_TOKENIZER_TYPE}"
    exit 1
fi

# training script command
CMD=""
if [ ! -z "$QNPU_DIR" ]; then
    CMD="source ${QNPU_DIR}/activate ;"
fi

CMD="${CMD} \
    cd $MODEL_DIR && \
    python3 -u ./pretrain_llama.py \
    --deepspeed \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --position-embedding-type $POS_EMB_TYPE \
    --no-bias \
    --layernorm-type rmsnorm \
    --activation-func-type swiglu \
    --layernorm-epsilon 1e-6 \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --num-attention-heads $NHEADS \
    --num-key-value-heads $N_KVHEADS \
    --seq-length $SEQ_LEN \
    --micro-batch-size ${MICRO_BATCH} \
    --global-batch-size ${GLOBAL_BATCH} \
    --train-iters 500000 \
    --log-interval ${LOG_INTERVAL} \
    --eval-iters 100 \
    --eval-interval 1000 \
    ${DATA_ARGS} \
    --optimizer ${OPTIMIZER} \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-5 \
    --lr 1.5e-4 \
    --min-lr 1.5e-5 \
    --lr-decay-style cosine \
    --lr-warmup-iters 2000 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --attention-dropout ${DROPOUT} \
    --hidden-dropout ${DROPOUT} \
    --tensorboard-dir $TENSORBOARD_DIR \
    --log-validation-ppl-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-timers-to-tensorboard \
    --load $CHECKPOINTS_DIR \
    --deepspeed_config=$DS_CONFIG  \
    --zero-stage=$ZERO_STAGE \
    --exit-interval $EXIT_INTERVAL \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --no-bias-dropout-fusion \
    --no-query-key-layer-scaling \
    --use-torch-compile $USE_TORCH_COMPILE \
    --use-fused-sdpa $USE_FUSED_SDPA \
    --use-fused-sdpa-with-recompute $USE_FUSED_SDPA_WITH_RECOMPUTE \
    $KILL_SWITCH_ARG \
    --bf16"

if [ $USE_HPU -eq 1 ]
then
    CMD="${CMD} --use_hpu --distributed-backend=hccl"
    CMD="${CMD} --hpu-deterministic"
fi

if [ ! -z "$DATA_IDX_DIR" ]; then
    CMD="${CMD} --data-idx-path ${DATA_IDX_DIR}"
fi

if [ ! "$NO_PIPELINE_PARALLEL" -eq 0 ]
then
    CMD="${CMD} --no-pipeline-parallel"
fi

if [ $SEQ_PARALLEL -eq 1 ]
then
    CMD="${CMD} --sequence-parallel"
fi

if [ $UNIV_CP -eq 1 ]
then
    echo "Loading Universal Checkpoint from ${CHECKPOINTS_DIR}"
    CMD="${CMD} --universal-checkpoint"
fi

if [ $CHECKPOINT_SAVE -eq 1 ]
then
    mkdir -p ${CHECKPOINTS_DIR}
    CMD="${CMD} --save $CHECKPOINTS_DIR --save-interval $SAVE_INTERVAL --verify-checkpoint --verify-checkpoint-model-type LLAMA"
fi

if [ $CKP_ACT -eq 1 ]
then
    CMD="${CMD} --checkpoint-activations --deepspeed-activation-checkpointing"
elif [ $CKP_ACT -eq 2 ]
then
    CMD="${CMD} --checkpoint-activations --deepspeed-activation-checkpointing --checkpoint-activations-granularity selective"
fi

if [ ! -z "$PROFILE" ]; then
    CMD="${CMD} --profile ${PROFILE}"
fi

if [ ! -z "$QNPU_DIR" ]; then
    rm -rf $HOME/.deepspeed_env
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> $HOME/.deepspeed_env
fi

# run!
deepspeed --num_nodes ${NUM_NODES} \
          --num_gpus ${NGPU_PER_NODE} \
          --no_local_rank \
          --no_python \
          $MULTINODE_CMD \
          /usr/bin/bash -c "$CMD" #2>&1 | tee ${OUTPUT_DIR}/log.txt
