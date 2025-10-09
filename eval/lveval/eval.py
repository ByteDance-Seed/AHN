import os
import re
import json
import argparse
import pandas as pd


from config import DATASET_METRIC
from utils import ensure_dir


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default=None)
    return parser.parse_args(args)

def custom_sort(s):
    letters = re.findall('[a-zA-Z]+', s)
    numbers = re.findall('\d+', s)
    return (letters, int(numbers[0])) if numbers else (letters, 0)

def scorer(dataset, predictions, answers, gold_anss):
    dataset_name = re.split('_.{1,3}k', dataset)[0]
    total_score = 0.
    total_sample = 0
    scores = {DATASET_METRIC[dataset_name].__name__: []}
    for (prediction, ground_truths, gold_ans) in zip(predictions, answers, gold_anss):
        total_sample += 1
        score = 0.
        for ground_truth in ground_truths:
            score = max(score, DATASET_METRIC[dataset_name](prediction, ground_truth, gold_ans))
            break
        total_score += score
        scores[DATASET_METRIC[dataset_name].__name__].append(score)
    return round(100 * total_score / total_sample, 2), scores

if __name__ == '__main__':
    args = parse_args()
    path = args.input_dir.rstrip("/")
    save_dir = f"{path}/eval_result/"
    ensure_dir(save_dir)
    

    all_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    all_files.sort(key=custom_sort)

    all_results = dict()
    all_scores = dict()
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, gold_anss, datas = [], [], [], []
        dataset = filename.split('.')[0]
        dataset_name = re.split('_.{1,3}k', dataset)[0]
        length = dataset.split('_')[-1]
        with open(f"{path}/{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                datas.append(data)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                gold_ans = data['gold_ans'] if 'gold_ans' in data else None
                gold_anss.append(gold_ans)
        score_mean, metric_scores = scorer(dataset, predictions, answers, gold_anss)
        all_scores[dataset] = score_mean
        if dataset_name in all_results:
            all_results[dataset_name].append({length: score_mean})
        else:
            all_results[dataset_name] = [{length: score_mean}]

    out_path = os.path.join(save_dir, "result.json")
    with open(out_path, "w") as f:
        json.dump(all_scores, f, ensure_ascii=False, indent=4)

    panda_list = []
    for dataset_name, length_score_list in all_results.items():
        lengths_scores = dict()
        for item in length_score_list:
            length, score = list(item.items())[0]
            lengths_scores[length] = score
        panda_dict = dict()
        panda_dict["dataset_name"] = dataset_name
        panda_dict.update(**lengths_scores)
        panda_list.append(panda_dict)
    dataframe = pd.DataFrame(panda_list)
    print(dataframe, '\n')
    out_path = os.path.join(save_dir, "result.csv")
    dataframe.to_csv(out_path, index=False)