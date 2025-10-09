# Copyright 2024 the LVEval team.
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT Licence

# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under MIT License, with the full license text
# available at https://github.com/infinigence/LVEval/blob/main/LICENSE.
#
# This modified file is released under the same license.

import os
import re
import torch
import argparse
from tqdm import tqdm

from config import (
    DATASET_MAXGEN, 
    DATASET_PROMPT, 
    DATASET_SELECTED, 
    DATASET_LENGTH_LEVEL,
)
from utils import (
    ensure_dir, 
    seed_everything,
    get_dataset_names,
    build_chat,
    post_process,
    model_generate,
    truncate_prompt,
    dump_preds_results,
    load_LVEval_dataset,
    load_model_and_tokenizer_once,
)

G_MODEL = None
G_TOKENIZER = None
G_DEVICE = None

def _init_worker(args):
    # worker index -> 0-based rank
    import multiprocessing as mp, torch
    rank = mp.current_process()._identity[0] - 1  # Pool gives 1..N
    torch.cuda.set_device(rank)
    if hasattr(torch, "set_default_device"):
        torch.set_default_device(f"cuda:{rank}")
    # load model/tokenizer ON THIS WORKER & DEVICE
    method, start_size, recent_size = args.method, args.start_size, args.recent_size
    model, tokenizer = load_model_and_tokenizer_once(
        rank, args.model_path, method=method, start_size=start_size, recent_size=recent_size
    )
    # Optional: compile here, not in parent
    # model = torch.compile(model)
    global G_MODEL, G_TOKENIZER, G_DEVICE
    G_MODEL, G_TOKENIZER = model, tokenizer
    G_DEVICE = torch.device(f"cuda:{rank}")

def get_pred(
    model,
    tokenizer,
    data,
    max_length,
    max_gen,
    prompt_format,
    model_name,
):
    preds = []
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        prompt = truncate_prompt(tokenizer, prompt, max_length)

        prompt = build_chat(tokenizer, prompt, model_name)
        pred = model_generate(tokenizer, prompt, max_gen, model)

        pred = post_process(pred, model_name)

        preds.append({
            "pred": pred,
            "answers": json_obj["answers"],
            "gold_ans": json_obj["answer_keywords"] if "answer_keywords" in json_obj else None,
            "input": json_obj["input"],
            "all_classes": json_obj["all_classes"] if "all_classes" in json_obj else None,
            "length": json_obj["length"],
        })
    return preds

def evaluate(mix):
    # mix only contains pure-Python data: shard, dataset name, formatting, etc.
    dataset_name = re.split('_.{1,3}k', mix["dataset"])[0]
    prompt_format = mix["dataset_prompt"][dataset_name]
    max_gen = mix["dataset_maxgen"][dataset_name]

    # Use globally-initialized model/tokenizer
    model = G_MODEL
    tokenizer = G_TOKENIZER

    preds = get_pred(
        model=model,
        tokenizer=tokenizer,
        data=mix["datas"],
        max_length=mix["max_length"],
        max_gen=max_gen,
        prompt_format=prompt_format,
        model_name=mix["model_name"],
    )
    return (mix["id"], preds)  # return, don't stuff CUDA things into shared dict

def split_datasets(input_list, num_parts, dataset, args):
    avg = len(input_list) // num_parts
    remainder = len(input_list) % num_parts
    result, start, id = [], 0, 0
    for i in range(num_parts):
        end = start + avg + (1 if i < remainder else 0)
        result.append({
            "datas": input_list[start:end],
            "dataset": dataset,
            "id": id,
            "model_path": args.model_path,
            "model_name": args.model_name,
            "max_length": args.model_max_length,
            "data_path": args.data_path,
            "out_dir": args.output_dir,
            "dataset_prompt": DATASET_PROMPT,
            "dataset_maxgen": DATASET_MAXGEN,
            "args": args,
        })
        start, id = end, id + 1
    return result

def multiple_processing_once(num_gpus, dataset, shared_dict, device_dict, args):
    datas = load_LVEval_dataset(dataset, args.data_path)
    mixs = split_datasets(datas, num_gpus, dataset, shared_dict, device_dict, args)
    ctx = torch.multiprocessing.get_context("spawn")
    pool = ctx.Pool(processes=num_gpus)
    pool.map(evaluate, mixs)
    pool.close()
    pool.join()
    merged_results = []
    for i in range(len(shared_dict)):
        merged_results += shared_dict[i]
    
    if args.method == "full-attention":
        model_name = f"{args.model_name}_{args.method}"
    else:
        model_name = f"{args.model_name}_{args.method}_as{args.start_size}_sw{args.recent_size}"
    
    out_path = os.path.join(args.output_dir, model_name, dataset + ".jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    dump_preds_results(merged_results, out_path)

def load_model_and_tokenizer_serial(num_gpus, args):
    model_path, method, start_size, recent_size = args.model_path, args.method, args.start_size, args.recent_size
    device_dict = dict()
    for i in range(num_gpus):
        load_model_and_tokenizer_once(i, model_path, device_dict, method=method, start_size=start_size, recent_size=recent_size)
    return device_dict

def multiple_processing(datasets, args):
    num_gpus = torch.cuda.device_count()
    ctx = torch.multiprocessing.get_context("spawn")
    # init model ONCE per worker process
    with ctx.Pool(processes=num_gpus, initializer=_init_worker, initargs=(args,)) as pool:
        for dataset in tqdm(datasets):
            if args.method == "full-attention":
                model_name = f"{args.model_name}_{args.method}"
            else:
                model_name = f"{args.model_name}_{args.method}_as{args.start_size}_sw{args.recent_size}"
            
            out_path = os.path.join(args.output_dir, model_name, dataset + ".jsonl")
            
            if os.path.exists(out_path) and not args.force_rerun:
                continue
            else:
                os.makedirs(os.path.dirname(out_path), exist_ok=True)

            datas = load_LVEval_dataset(dataset, args.data_path)
            mixs = split_datasets(datas, num_gpus, dataset, args)

            # map returns list of (shard_id, preds)
            results = pool.map(evaluate, mixs)
            results.sort(key=lambda x: x[0])
            merged = []
            for _, preds in results:
                merged += preds
            
            dump_preds_results(merged, out_path)

def single_processing(datasets, args):
    model, tokenizer = load_model_and_tokenizer_once(-1, args.model_path)
    for dataset in tqdm(datasets):
        datas = load_LVEval_dataset(dataset, args.data_path)
        dataset_name = re.split('_.{1,3}k', dataset)[0]
        prompt_format = DATASET_PROMPT[dataset_name]
        max_gen = DATASET_MAXGEN[dataset_name]
        preds = get_pred(
            model,
            tokenizer,
            datas,
            args.model_max_length,
            max_gen,
            prompt_format,
            args.model_name,
        )
        dump_preds_results(preds, os.path.join(args.output_dir, dataset + ".jsonl"))

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None, required=True)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--model-max-length", type=int, default=15500)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--start_size", type=int, default=128)
    parser.add_argument("--recent_size", type=int, default=8064)
    parser.add_argument("--output-dir", type=str, default="./tmp/lveval")
    parser.add_argument("--single-process", action="store_true")
    parser.add_argument("--force_rerun", action="store_true")
    return process_args(parser.parse_args(args))

def process_args(args):
    model_path = args.model_path.rstrip("/")
    if not args.model_name:
        args.model_name = os.path.basename(model_path)
    return args



if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    ensure_dir(args.output_dir)
    datasets = get_dataset_names(DATASET_SELECTED, DATASET_LENGTH_LEVEL)

    if args.single_process:
        single_processing(datasets, args)
    else:
        multiple_processing(datasets, args)
