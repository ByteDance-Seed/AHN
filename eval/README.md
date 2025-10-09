# Evaluation

## 1. Install evaluation dependencies
```bash
pip install -e ".[eval]" 
```

## 2. Run inference and evaluation together
### LongBench
```bash
cd eval/longbench

MODEL_NAME=qwen2.5-3b-ahn-gdn # model identifier, the results will be save in $OUTPUT_DIR/$MODEL_NAME
MERGED_MODEL_PATH=../../merged_ckpt/Qwen-2.5-Instruct-3B-AHN-GDN # path to weights of base model and AHN
METHOD=ahn # inference method
MAX_INPUT_LENGTH=32000 # maximum input tokens
NUM_ATTENTION_SINK=128 # number of attention sink tokens
SLIDING_WINDOW=8064 # sliding window size
OUTPUT_DIR=./eval_results # directory to save predictions

bash eval.sh $MODEL_NAME $MERGED_MODEL_PATH $METHOD $MAX_INPUT_LENGTH $NUM_ATTENTION_SINK $SLIDING_WINDOW $OUTPUT_DIR
```

### LV-Eval
```bash
cd eval/lveval

MODEL_NAME=qwen2.5-3b-ahn-gdn # model identifier, the results will be save in $OUTPUT_DIR/$MODEL_NAME
MERGED_MODEL_PATH=../../merged_ckpt/Qwen-2.5-Instruct-3B-AHN-GDN # path to weights of base model and AHN
METHOD=ahn # inference method
MAX_INPUT_LENGTH=256000 # maximum input tokens
NUM_ATTENTION_SINK=128 # number of attention sink tokens
SLIDING_WINDOW=32640 # sliding window size
OUTPUT_DIR=./eval_results # directory to save predictions

python3 pred.py \
  --model-name $MODEL_NAME \
  --model-path $MERGED_MODEL_PATH \
  --method $METHOD \
  --model-max-len $MAX_INPUT_LENGTH \
  --start_size $NUM_ATTENTION_SINK \
  --recent_size $SLIDING_WINDOW \
  --output-dir $OUTPUT_DIR

python3 eval.py \
  --input-dir ${OUTPUT_DIR}/${MODEL_NAME}_${METHOD}_as${NUM_ATTENTION_SINK}_sw${SLIDING_WINDOW}
```

## 3. Running Inference and Evaluation Separately

### LongBench

#### Inference
```bash
cd eval/longbench

MODEL_NAME=qwen2.5-3b-ahn-gdn # model identifier, the results will be save in $OUTPUT_DIR/$MODEL_NAME
MERGED_MODEL_PATH=../../merged_ckpt/Qwen-2.5-Instruct-3B-AHN-GDN # path to weights of base model and AHN
METHOD=ahn # inference method
MAX_INPUT_LENGTH=32000 # maximum input tokens
NUM_ATTENTION_SINK=128 # number of attention sink tokens
SLIDING_WINDOW=8064 # sliding window size
DATASET=hotpotqa # one of ("dureader" "hotpotqa" "musique" "narrativeqa" "qmsum" "triviaqa")
OUTPUT_DIR=./eval_results # directory to save predictions

CUDA_VISIBLE_DEVICES=0 python pred.py \
  --model_name $MODEL_NAME \
  --model_path $MERGED_MODEL_PATH \
  --max_length $MAX_INPUT_LENGTH \
  --method $METHOD \
  --dataset $DATASET \
  --attention_sink $NUM_ATTENTION_SINK \
  --sliding_window $SLIDING_WINDOW \
  --results_dir $OUTPUT_DIR
```
#### Evaluation
```bash
python eval.py \
  --model $MODEL_NAME \
  --results_path $OUTPUT_DIR/${MODEL_NAME}_${MAX_INPUT_LENGTH}
```

### InfiniteBench
#### Inference
```bash
cd eval/longbench

MODEL_NAME=qwen2.5-3b-ahn-gdn # model identifier, the results will be save in $OUTPUT_DIR/$MODEL_NAME
MERGED_MODEL_PATH=../../merged_ckpt/Qwen-2.5-Instruct-3B-AHN-GDN # path to weights of base model and AHN
METHOD=ahn # inference method
MAX_INPUT_LENGTH=128000 # maximum input tokens
NUM_ATTENTION_SINK=128 # number of attention sink tokens
SLIDING_WINDOW=32640 # sliding window size
DATASET=infinitebench
SPLIT=longbook_qa_eng # or longbook_qa_chn
OUTPUT_DIR=./eval_results # directory to save predictions


CUDA_VISIBLE_DEVICES=0 python pred.py \
  --model_name $MODEL_NAME \
  --model_path $MERGED_MODEL_PATH \
  --max_length $MAX_INPUT_LENGTH \
  --method $METHOD \
  --dataset $DATASET \
  --split $SPLIT \
  --attention_sink $NUM_ATTENTION_SINK \
  --sliding_window $SLIDING_WINDOW \
  --results_dir $OUTPUT_DIR
```
#### Evaluation
```bash
python eval.py \
  --model $MODEL_NAME \
  --results_path $OUTPUT_DIR/${MODEL_NAME}_${MAX_INPUT_LENGTH}
```

We would like to thank the developers of [LongBench](https://github.com/THUDM/LongBench), [LVEval](https://github.com/infinigence/LVEval), and [InfiniteBench](https://github.com/OpenBMB/InfiniteBench) for their open-source contributions that support our evaluation. This repository retains only the minimal usable scripts for convenience; please refer to the original repositories for complete functionality and details.