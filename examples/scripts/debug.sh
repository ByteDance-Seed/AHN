# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -x

IFS=',' read -ra ALL_PORTS <<< $METIS_WORKER_0_PORT
export NCCL_SOCKET_IFNAME=eth0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT=${WANDB_PROJECT:-"AHN"}

export ALLOW_EXTRA_ARGS=1
export DISABLE_VERSION_CHECK=1

JOB_NAME=${JOB_NAME:-"qwen2.5-7b_ahn_gdn_debugging"}
OUTPUT_DIR=${OUTPUT_DIR:-"./tmp/ahn_gdn_debugging"}
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi


DISTRIBUTED_ARGS="
    --nnodes=$ARNOLD_WORKER_NUM \
    --node_rank=$ARNOLD_ID \
    --nproc_per_node=1 \
    --master_addr=$ARNOLD_WORKER_0_HOST \
    --master_port=${ALL_PORTS[0]}
  "

torchrun $DISTRIBUTED_ARGS src/train.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --stage sft \
    --finetuning_type freeze \
    --freeze_extra_modules ahn \
    --loss_type kl \
    --use_normalized_l2 True \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --do_train \
    --use_fast_tokenizer \
    --flash_attn fa2 \
    --sliding_window 512 \
    --sliding_window_type random \
    --ahn_position random \
    --layer_implementation Qwen2MemDecoderLayer \
    --ahn_implementation GatedDeltaNet \
    --dataset chatqa2 \
    --template qwen \
    --filter_len 528 \
    --cutoff_len 16384 \
    --max_samples 4096 \
    --overwrite_cache false \
    --overwrite_output_dir false \
    --preprocessing_num_workers 16 \
    --output_dir $OUTPUT_DIR \
    --save_ahn_only True \
    --logging_steps 1 \
    --save_steps 500 \
    --plot_loss \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --weight_decay 0.1 \
    --bf16 true \
    --report_to wandb \
    --run_name $JOB_NAME \
    --ddp_timeout 180000000 \