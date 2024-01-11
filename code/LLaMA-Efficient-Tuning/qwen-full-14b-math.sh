#!/bin/bash

# ================= 参数配置 ===============
# NCCL参数
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0,Bond1.2503
export NCCL_IB_HCA=mlx5_2,mlx5_5
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/


# 随机种子
export RANDOM_SEED=1234

# 数据与模型路径
# DATA_VERSION=data #文件夹

DATASET_NAME=TAL-SAQ5K-ans,math_train_summ,math_test_summ,ape210k,math23k,GSM8K
#math-cot-input,TAL-cot-input,TAL-SAQ5K-ans,math_train_summ,math_test_summ,ape210k,math23k,GSM8K

MODEL_NAME=14B-all #模型名称 
# MODEL_NAME=QWen # 模型名称
MODEL_PATH=/work/share/public/weights/Qwen-14B-0925/Qwen-14B-Chat
# MODEL_PATH=/work/share/public/weights/Qwen-72B-Chat
DATA_PATH=/work/cache/paper_zhuanli_game/aaai2024comp/train_dataset

# DATA_PATH=/work/home/project/LLaMA-Efficient-Tuning/data
DATASET_DIR=$DATA_PATH

# 获取脚本所在目录的绝对路径
SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")


# 配置和脚本路径
TRAIN_BASH_PY=${SCRIPTPATH}/src/train_bash.py

# 训练参数
STAGE=sft
MAX_SOURCE_LENGTH=2048
MAX_TARGET_LENGTH=2048
FINETUNING_TYPE=full
BATCH_SIZE=4
GRADIENT_STEPS=2
LR_SCHEDULER_TYPE=cosine
LOGGING_STEPS=10
SAVE_STEPS=-1
LEARNING_RATE=4e-5
NUM_EPOCHS=4
PROMPT_TEMPLATE=chatml
NNODES=5 # 节点个数
NPROC=8 # 一个节点有几张GPU

# 输出与日志路径
LOG_PATH=${SCRIPTPATH}/logs
EXP_NAME=$(date +"%m%d-%H")-${NNODES}nodes
OUTPUT_PATH=/work/cache/model/aaai24-math/${MODEL_NAME}
LOG_DIR=$LOG_PATH/${EXP_NAME}
OUTPUT_DIR=$OUTPUT_PATH/${EXP_NAME}


# ========================================

# 创建必要的目录
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}
cp -r ${SCRIPTPATH}/qwen-full-14b-math.sh ${OUTPUT_DIR}
cp -r ${SCRIPTPATH}/deep_speed.json ${OUTPUT_DIR}


# 启动训练
WORLD_SIZE=8  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${NPROC} \
    --master_port=${MASTER_PORT} \
    --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} \
    ${TRAIN_BASH_PY} \
    --deepspeed ${SCRIPTPATH}/deep_speed.json \
    --stage ${STAGE} \
    --model_name_or_path ${MODEL_PATH} \
    --do_train \
    --dataset ${DATASET_NAME} \
    --dataset_dir ${DATASET_DIR}\
    --max_source_length ${MAX_SOURCE_LENGTH} \
    --max_target_length ${MAX_TARGET_LENGTH} \
    --finetuning_type ${FINETUNING_TYPE} \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_cache \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_STEPS} \
    --lr_scheduler_type ${LR_SCHEDULER_TYPE} \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${NUM_EPOCHS} \
    --plot_loss \
    --bf16 \
    --template ${PROMPT_TEMPLATE} \
    --weight_decay 0.01 \
    --overwrite_output_dir \
    2>&1 | tee ${LOG_DIR}/${RANK}-${HOSTNAME}.log
