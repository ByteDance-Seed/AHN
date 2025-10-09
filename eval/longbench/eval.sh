# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT Licence

#!/bin/bash
# Usage check
if [ "$#" -lt 7 ]; then
    echo "Usage: $0 MODEL_NAME CKPT_PATH METHOD MAX_LENGTH ATTENTION_SINK SLIDING_WINDOW OUTPUT_DIR"
    exit 1
fi

# Read input arguments
model_name=$1
ckpt_path=$2
method=$3
max_length=$4
attention_sink=$5
sliding_window=$6


# Define output directory
output_dir=$7

# Define evaluation tasks
# eval_sets=("narrativeqa" "qasper" "multifieldqa_en" "multifieldqa_zh" "hotpotqa" "2wikimqa" "musique" \
#            "dureader" "gov_report" "qmsum" "multi_news" "vcsum" "trec" "triviaqa" "lsht" \
#            "passage_count" "passage_retrieval_en" "passage_retrieval_zh" "lcc" "repobench-p" "samsum")
eval_sets=("dureader" "hotpotqa" "musique" "narrativeqa" "qmsum" "triviaqa") # average sequence length > 8k
datasets-cli test zai-org/LongBench --all_configs

# Number of GPUs to use
NUM_GPUS=$(nvidia-smi -L | wc -l)

# Initialize arrays to track tasks and GPUs
declare -A gpu_task      # Maps GPU ID to currently running task name
declare -A gpu_pid       # Maps GPU ID to currently running process ID

# Initialize task counters
task_idx=0
completed_tasks=0
total_tasks=${#eval_sets[@]}

echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting execution with $NUM_GPUS GPUs for $total_tasks tasks"

# First, launch tasks on all available GPUs
for ((gpu=0; gpu<NUM_GPUS && task_idx<total_tasks; gpu++)); do
    task=${eval_sets[$task_idx]}
    
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Launching task '$task' on GPU $gpu"
    
    CUDA_VISIBLE_DEVICES=$gpu python pred.py \
        --dataset "$task" \
        --model_name "$model_name" \
        --model_path "$ckpt_path" \
        --max_length "$max_length" \
        --method "$method" \
        --attention_sink "$attention_sink" \
        --sliding_window "$sliding_window" \
        --results_dir "$output_dir" &
    
    # Store the process ID
    pid=$!
    gpu_pid[$gpu]=$pid
    gpu_task[$gpu]=$task
    
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Task '$task' started with PID $pid on GPU $gpu"
    
    ((task_idx++))
done

# Then, continuously check for completed tasks and start new ones
while [ $completed_tasks -lt $total_tasks ]; do
    for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
        # Skip if no task was assigned to this GPU
        [ -z "${gpu_pid[$gpu]}" ] && continue
        
        # Check if the task on this GPU has completed
        if ! kill -0 ${gpu_pid[$gpu]} 2>/dev/null; then
            completed_task=${gpu_task[$gpu]}
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Task '$completed_task' completed on GPU $gpu"
            
            # Mark task as completed
            ((completed_tasks++))
            
            # Clear the GPU's assignment
            unset gpu_task[$gpu]
            unset gpu_pid[$gpu]
            
            # If there are more tasks to run, assign one to this GPU
            if [ $task_idx -lt $total_tasks ]; then
                task=${eval_sets[$task_idx]}
                
                echo "$(date '+%Y-%m-%d %H:%M:%S') - Launching task '$task' on GPU $gpu"
                
                CUDA_VISIBLE_DEVICES=$gpu python pred.py \
                    --dataset "$task" \
                    --model_name "$model_name" \
                    --model_path "$ckpt_path" \
                    --max_length "$max_length" \
                    --method "$method" \
                    --attention_sink "$attention_sink" \
                    --sliding_window "$sliding_window" \
                    --results_dir "$output_dir" &
                
                # Store the process ID
                pid=$!
                gpu_pid[$gpu]=$pid
                gpu_task[$gpu]=$task
                
                echo "$(date '+%Y-%m-%d %H:%M:%S') - Task '$task' started with PID $pid on GPU $gpu"
                
                ((task_idx++))
            fi
        fi
    done
    
    # Progress report
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Progress: $completed_tasks/$total_tasks tasks completed"
    
    # Sleep to avoid excessive CPU usage
    sleep 10
done

echo "$(date '+%Y-%m-%d %H:%M:%S') - All tasks completed! Running evaluation..."

# Run evaluation
results_path="${output_dir}/${model_name}_${max_length}"
python eval.py --model "${model_name}_${max_length}" --results_path "$results_path"

echo "$(date '+%Y-%m-%d %H:%M:%S') - Evaluation complete"