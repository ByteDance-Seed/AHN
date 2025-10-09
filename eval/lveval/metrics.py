# Copyright 2024 the LVEval team.
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT Licence

# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under MIT License, with the full license text
# available at https://github.com/infinigence/LVEval/blob/main/LICENSE.
#
# This modified file is released under the same license.


"""Functions for computing metrics.

Part of following code are modified from `https://github.com/THUDM/LongBench/blob/a80fd111d6e5fe1735eb7be53fece976706f8e0c/metrics.py`
"""


import re
import jieba
import string
from rouge import Rouge
from collections import Counter



ABANDON_WORDS_EN = ['and', 'to', 'of', 'in', 'her', 'was', 'with', 'for', 'it', 'from', 'is', 'that', 'his', 'he', 'by', 'she', 'they', 'or', 'at', 'because', 'be', 'on', 'are', 'their', 'what', 'as', 'had', 'were', 'about', 'being', 'this', 'who', 'but', 'have', 'has', 'when', 'which', 'does']
ABANDON_WORDS_ZH = ['的', '和', '是', '等', '在', '年', '可以', '为', '与', '‰', '了', '或', '一种', '月', 'c', '至', '日', '有', '进行', '于', '不', '中', '×', '根据', '小', '由', '亩', '也', '要', '指', '法', '会', '元', '主要', '以及', '通过', '首先', '对', '然后', '号', '以', '所', '后', '丁', '包括', '无', '将', '用', '能', '形', '方面', '因素', '位于', '而', '从', '到', '一定', '用于', '但', '使用', '让', '具有', '并', '亿元', '万元', '上', '类', '基于', '才', '来', '地', '片', '其他', '个', '或者', '变得', '时', '给', '你', '使', '条', '受', '已经', '带', '度']

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s):
    """Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))
    
def rouge_score(prediction, ground_truth, gold_ans=None, **kwargs):
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except:
        return 0.0
    return scores["rouge-l"]["f"]

def rouge_zh_score(prediction, ground_truth, gold_ans=None, **kwargs):
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False))) 
    score = rouge_score(prediction, ground_truth)
    return score

def rouge_zh_score_blacklist(prediction, ground_truth, gold_ans=None, **kwargs):
    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False))) 
    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens = [normalize_zh_answer(token) for token in prediction_tokens]
    ground_truth_tokens = [normalize_zh_answer(token) for token in ground_truth_tokens]
    filtered_prediction_tokens = [i for i in prediction_tokens if i not in ABANDON_WORDS_ZH]
    filtered_ground_truth_tokens = [i for i in ground_truth_tokens if i not in ABANDON_WORDS_ZH]
    prediction = " ".join(filtered_prediction_tokens)
    ground_truth = " ".join(filtered_ground_truth_tokens) 
    score = rouge_score(prediction, ground_truth)
    return score

def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def qa_f1_score(prediction, ground_truth, gold_ans=None, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)

def qa_f1_score_factrecall(prediction, ground_truth, gold_ans=None, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    recall = 1.0 * num_same / len(ground_truth_tokens)

    return recall

def qa_f1_score_with_gold_ans(prediction, ground_truth, gold_ans=None, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    # answer keywords recall
    if gold_ans:
        gold_ans_tokens = normalize_answer(gold_ans)
        gold_ans_tokens = gold_ans_tokens.split()
        common = Counter(prediction_tokens) & Counter(gold_ans_tokens)
        filtered_common = {key: value for key, value in common.items() if key not in ABANDON_WORDS_EN}
        num_same = sum(filtered_common.values())
        recall = 1.0 * num_same / len(gold_ans_tokens)
        if recall < 0.2: return 0.

    return f1_score(prediction_tokens, ground_truth_tokens)

def qa_f1_zh_score(prediction, ground_truth, gold_ans=None, **kwargs):
    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens = [normalize_zh_answer(token) for token in prediction_tokens]
    ground_truth_tokens = [normalize_zh_answer(token) for token in ground_truth_tokens]
    prediction_tokens = [token for token in prediction_tokens if len(token) > 0]
    ground_truth_tokens = [token for token in ground_truth_tokens if len(token) > 0]
    return f1_score(prediction_tokens, ground_truth_tokens)

def qa_f1_zh_score_factrecall(prediction, ground_truth, gold_ans=None, **kwargs):
    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens = [normalize_zh_answer(token) for token in prediction_tokens]
    ground_truth_tokens = [normalize_zh_answer(token) for token in ground_truth_tokens]
    prediction_tokens = [token for token in prediction_tokens if len(token) > 0]
    ground_truth_tokens = [token for token in ground_truth_tokens if len(token) > 0]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return recall

def qa_f1_zh_score_with_gold_ans(prediction, ground_truth, gold_ans=None, **kwargs):
    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens = [normalize_zh_answer(token) for token in prediction_tokens]
    ground_truth_tokens = [normalize_zh_answer(token) for token in ground_truth_tokens]
    prediction_tokens = [token for token in prediction_tokens if len(token) > 0]
    ground_truth_tokens = [token for token in ground_truth_tokens if len(token) > 0]
    # answer keywords recall
    if not gold_ans:
        gold_ans = ground_truth
    if gold_ans:
        gold_ans_tokens = list(jieba.cut(gold_ans, cut_all=False))
        gold_ans_tokens = [normalize_zh_answer(token) for token in gold_ans_tokens]
        gold_ans_tokens = [token for token in gold_ans_tokens if len(token) > 0]
        common = Counter(prediction_tokens) & Counter(gold_ans_tokens)
        filtered_common = {key: value for key, value in common.items() if key not in ABANDON_WORDS_ZH}
        num_same = sum(filtered_common.values())
        recall = 1.0 * num_same / len(gold_ans_tokens)
        if recall < 0.4: return 0.

    return f1_score(prediction_tokens, ground_truth_tokens)