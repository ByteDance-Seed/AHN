# Copyright 2024 the MemoryLLM team.
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT Licence

# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under MIT License, with the full license text
# available at https://github.com/wangyu-ustc/MemoryLLM/blob/main/LICENSE.
#
# This modified file is released under the same license.

import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
import time
import shutil


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    parser.add_argument("--max_length", default=None, type=int)
    parser.add_argument("--split", default="longbook_qa_eng", type=str)
    parser.add_argument("--split_model", default=False, action="store_true")
    parser.add_argument("--part", default=0, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--dataset", default=None, type=str)
    parser.add_argument("--retrieval", default=None, help="Retrieval method", type=str)
    parser.add_argument("--exclude_or", default=False, action="store_true")
    parser.add_argument("--force_run", default=False, action="store_true")
    parser.add_argument("--method", type=str, default="qwen2")
    parser.add_argument("--attention_sink", type=int, default=128)
    parser.add_argument("--sliding_window", type=int, default=512)
    parser.add_argument("--enable_yarn", default=False, action="store_true")
    parser.add_argument("--fa_layer_ids", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    return parser.parse_args(args)  # return parser.parse_known_args(args)[0]


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "qwen2" in model_name:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    elif "qwen3" in model_name:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    elif "llama2" in model_name and "chat" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"

    return prompt


def post_process(response, model_name):
    """
    Post-process the response.

    Args:
        response (str): Model response.
        model_name (str): Model name.

    Returns:
        str: Post-processed response.
    """
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def get_pred(
    model,
    tokenizer,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    device,
    model_name,
    retrieval=None,
    exclude_or=False,
):

    preds = []

    count = 0
    for json_obj in tqdm(data):

        count += 1

        if exclude_or:
            if "or" in json_obj["input"]:
                continue

        prompt = prompt_format.format(**json_obj)

        if retrieval is None:
            # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
            tokenized_prompt = tokenizer(
                prompt, truncation=False, return_tensors="pt"
            ).input_ids[0]
        else:
            prompt_context = prompt.split(
                prompt_format.split("{context}")[-1].split("Question")[0]
            )[0]
            tokenized_prompt = tokenizer(
                prompt_context, truncation=False, return_tensors="pt"
            ).input_ids[0]

        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(
                prompt, truncation=False, return_tensors="pt", add_special_tokens=False
            ).input_ids[0]

        if max_length > 0 and len(tokenized_prompt) > max_length:
            if retrieval is None:
                # truncate at the beginning:
                prompt = tokenizer.decode(
                    tokenized_prompt[-(max_length - max_gen) :],
                    skip_special_tokens=True,
                )

            else:
                tokenized_prompt = tokenizer(
                    tokenizer.decode(
                        tokenized_prompt[-(max_length - max_gen) :],
                        skip_special_tokens=True,
                    ),
                    truncation=False,
                    return_tensors="pt",
                    add_special_tokens=False,
                ).input_ids[0]

        if dataset not in [
            "trec",
            "triviaqa",
            "samsum",
            "lsht",
            "lcc",
            "repobench-p",
        ]:  # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        if "chatglm3" in model_name:
            input = prompt
        else:
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)

        with torch.inference_mode():
            context_length = input.input_ids.shape[-1]

            if "qwen3" in model_name:
                # qwen3-non-thinking
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=200,
                    min_p=0.0,
                )[0]
            else:
                output = model.generate(
                    **input,
                    max_new_tokens=max_gen,
                    num_beams=1,
                    do_sample=False,
                    temperature=1.0,
                )[0]

        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)

        try:
            print(f"Prediction: {pred}, Answer: {json_obj['answers']}")
            preds.append(
                {
                    "pred": pred,
                    "answers": json_obj["answers"],
                    "all_classes": json_obj["all_classes"],
                    "length": json_obj["length"],
                }
            )
        except:
            print(f"Prediction: {pred}, Answer: {json_obj['answer']}")
            preds.append({"pred": pred, "answers": json_obj["answer"]})
    return preds


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path, device):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    model = model.eval()
    return model, tokenizer


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model_name

    if args.method == "ahn":
        # register AHN modules
        from ahn.transformer.qwen2_ahn import register_customized_qwen2
        from ahn.transformer.qwen3_ahn import register_customized_qwen3

        register_customized_qwen2()
        register_customized_qwen3()

    model, tokenizer = load_model_and_tokenizer(args.model_path, device)

    if args.enable_yarn:
        model.config.rope_scaling = {
            "type": "yarn",
            "factor": 4,
            "original_max_position_embeddings": 32768,
        }

    if args.method == "ahn":
        model.config.sliding_window = args.sliding_window
        if args.fa_layer_ids is not None and args.fa_layer_ids != "":
            model.config.fa_layer_ids = args.fa_layer_ids
        model.config.num_attn_sinks = args.attention_sink
    elif args.method == "window-attention":
        model.config.use_sliding_window = True
        model.config.max_window_layers = -1
        model.config.sliding_window = args.sliding_window

    if args.e:
        datasets = [
            "qasper",
            "multifieldqa_en",
            "hotpotqa",
            "2wikimqa",
            "gov_report",
            "multi_news",
            "trec",
            "triviaqa",
            "samsum",
            "passage_count",
            "passage_retrieval_en",
            "lcc",
            "repobench-p",
        ]
    else:
        if args.dataset is None:
            datasets = [
                "narrativeqa",
                "qasper",
                "multifieldqa_en",
                "multifieldqa_zh",
                "hotpotqa",
                "2wikimqa",
                "musique",
                "dureader",
                "gov_report",
                "qmsum",
                "multi_news",
                "vcsum",
                "trec",
                "triviaqa",
                "lsht",
                "passage_count",
                "passage_retrieval_en",
                "passage_retrieval_zh",
                "lcc",
                "repobench-p",
                "samsum",
            ]
        else:
            datasets = [args.dataset]

    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("configs/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("configs/dataset2maxlen.json", "r"))
    # predict on each dataset
    # if not os.path.exists(f"longbench/pred_seed{args.seed}"):
    #     os.makedirs(f"longbench/pred_seed{args.seed}")
    # if not os.path.exists(f"longbench/pred_seed{args.seed}_e"):
    #     os.makedirs(f"longbench/pred_seed{args.seed}_e")

    for dataset in datasets:
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        
        if args.e:
            data = load_dataset(
                "zai-org/LongBench", f"{dataset}_e", split="test", trust_remote_code=True
            )
            data.save_to_disk(f"longbench/data_e/{dataset}")

            if not os.path.exists(f"longbench/pred_seed{args.seed}_e/{model_name}"):
                os.makedirs(f"longbench/pred_seed{args.seed}_e/{model_name}")
            out_path = f"longbench/pred_seed{args.seed}_e/{model_name}/{dataset}.jsonl"
            # save dataset
        else:
            if "infinitebench" in dataset:
                from datasets import Value, Sequence, Features

                ft = Features(
                    {
                        "id": Value("int64"),
                        "context": Value("string"),
                        "input": Value("string"),
                        "answer": Sequence(Value("string")),
                        "options": Sequence(Value("string")),
                    }
                )
                data = load_dataset(
                    "xinrongzhang2022/InfiniteBench",
                    features=ft,
                    trust_remote_code=True,
                )[args.split]
                dataset = f"{dataset}_{args.split}"
            else:
                data = load_dataset(
                    "zai-org/LongBench", dataset, split="test", trust_remote_code=True
                )

            if args.results_dir is not None:
                results_dir = os.path.join(
                    args.results_dir, f"{args.model_name}_{args.max_length}"
                )
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)
                out_path = os.path.join(results_dir, f"{dataset}.jsonl")
            else:
                results_dir = args.results_dir
                if not os.path.exists(
                    f"eval_results/longbench/pred_seed{args.seed}/{args.model_name}_{args.max_length}"
                ):
                    os.makedirs(
                        f"eval_results/longbench/pred_seed{args.seed}/{args.model_name}_{args.max_length}"
                    )
                out_path = f"eval_results/longbench/pred_seed{args.seed}/{args.model_name}_{args.max_length}/{dataset}.jsonl"

        if args.exclude_or:
            out_path = out_path.split("/")
            out_path[-2] += "_exor"
            out_path = "/".join(out_path)

            if not os.path.exists(os.path.dirname(out_path)):
                os.makedirs(os.path.dirname(out_path))

        if os.path.exists(out_path) and not args.force_run:
            continue

        preds = get_pred(
            model,
            tokenizer,
            data,
            args.max_length,
            max_gen,
            prompt_format,
            dataset,
            device,
            model_name,
            args.retrieval,
            exclude_or=args.exclude_or,
        )

        tmp_path = f"/tmp/{dataset}.jsonl"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write("\n")

        time.sleep(1)

        shutil.copy(tmp_path, out_path)
