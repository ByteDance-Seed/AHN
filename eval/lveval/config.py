# Copyright 2024 the LVEval team.
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT Licence

# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under MIT License, with the full license text
# available at https://github.com/infinigence/LVEval/blob/main/LICENSE.
#
# This modified file is released under the same license.


from metrics import (
    qa_f1_score,
    qa_f1_score_with_gold_ans,
    qa_f1_zh_score,
    qa_f1_zh_score_with_gold_ans,
    rouge_zh_score_blacklist,
)

DATASET_MAXGEN = {
    "hotpotwikiqa_mixup": 64,
    "loogle_SD_mixup": 64,
    "loogle_CR_mixup": 64,
    "loogle_MIR_mixup": 64,
    "multifieldqa_en_mixup": 64,
    "multifieldqa_zh_mixup": 64,
    "factrecall_en": 16,
    "factrecall_zh": 16,
    "cmrc_mixup": 64,
    "lic_mixup": 64,
    "dureader_mixup": 64,
}

DATASET_PROMPT = {
    "hotpotwikiqa_mixup": "Answer the question based on the given passages. Questions and answers are only relevant to some passages. Only give me the answer and do not output any other explanation and evidence.\n\nArticle: {context}\n\nPlease answer the following question based on the above passages. Questions and answers are only relevant to some passages. Only give me the answer and do not output any other explanation and evidence.\n\nQuestion: {input}\nAnswer:",    
    "loogle_SD_mixup": "Please answer the following question based on the given passages. Questions and answers are only relevant to one passage. Only give me the answer and do not output any other explanation and evidence.\n\nArticle: {context}\n\nPlease answer the following question based on the above passages. Questions and answers are only relevant to one passage. Only give me the answer and do not output any other explanation and evidence.\n\nQuestion: {input}\nAnswer:",
    "loogle_CR_mixup": "Please answer the following question based on the given passages. Questions and answers are only relevant to one passage. Only give me the answer and do not output any other explanation and evidence.\n\nArticle: {context}\n\nPlease answer the following question based on the above passages. Questions and answers are only relevant to one passage. Only give me the answer and do not output any other explanation and evidence.\n\nQuestion: {input}\nAnswer:",
    "loogle_MIR_mixup": "Please answer the following question based on the given passages. Questions and answers are only relevant to one passage. Only give me the answer and do not output any other explanation and evidence.\n\nArticle: {context}\n\nPlease answer the following question based on the above passages. Questions and answers are only relevant to one passage. Only give me the answer and do not output any other explanation and evidence.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_en_mixup": "Please answer the following question based on the given passages. Questions and answers are only relevant to one passage. Only give me the answer and do not output any other explanation and evidence.\n\nArticle: {context}\n\nPlease answer the following question based on the above passages. Questions and answers are only relevant to one passage. Only give me the answer and do not output any other explanation and evidence.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh_mixup": "请阅读以下文章并用中文回答问题，问题和答案只与其中一篇文章有关。只需要直接给出问题的答案，不要输出其他任何解释和证据。\n\n文章：{context}\n\n请基于上面的文章回答下面的问题，问题和答案只与其中一篇文章有关。只需要直接给出问题的答案，不要输出其他任何解释和证据。\n\n问题：{input}\n回答：",
    "factrecall_en": "Please answer the following questions based on the given article.\n\nArticle: {context}\n\nPlease answer the following questions based on the above article.\n\nQuestion: {input}\nAnswer:",
    "factrecall_zh": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n现在请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "cmrc_mixup": "请根据下面给定的文章回答问题，问题和答案只与其中一篇文章有关。\n\n文章：{context}\n\n现在请基于上述文章回答下面的问题，问题和答案只与其中一篇文章有关。\n\n问题：{input}\n回答：",
    "lic_mixup": "请根据下面给定的文章回答问题，问题和答案只与其中一篇文章有关。\n\n文章：{context}\n\n请现在基于上述文章回答下面的问题，问题和答案只与其中一篇文章有关。\n\n问题：{input}\n回答：",
    "dureader_mixup": "请根据下面给定的文章回答问题，问题和答案只与其中一篇文章有关。\n\n文章：{context}\n\n现在请基于上述文章回答下面的问题，问题和答案只与其中一篇文章有关。\n\n问题：{input}\n回答：",
}

DATASET_METRIC = {
    "hotpotwikiqa_mixup": qa_f1_score_with_gold_ans,
    "loogle_SD_mixup": qa_f1_score_with_gold_ans,
    "loogle_CR_mixup": qa_f1_score_with_gold_ans,
    "loogle_MIR_mixup": qa_f1_score_with_gold_ans,
    "multifieldqa_en_mixup": qa_f1_score_with_gold_ans,
    "multifieldqa_zh_mixup": qa_f1_zh_score_with_gold_ans,
    "factrecall_en": qa_f1_score,
    "factrecall_zh": qa_f1_zh_score,
    "cmrc_mixup": qa_f1_zh_score_with_gold_ans,
    "lic_mixup": qa_f1_zh_score_with_gold_ans,
    "dureader_mixup": rouge_zh_score_blacklist,
}

DATASET_SELECTED = [
    "hotpotwikiqa_mixup",
    "loogle_SD_mixup",
    "loogle_CR_mixup",
    "loogle_MIR_mixup",
    "multifieldqa_en_mixup",
    "multifieldqa_zh_mixup",
    "factrecall_en",
    "factrecall_zh",
    "cmrc_mixup",
    "lic_mixup",
    "dureader_mixup",
]

DATASET_LENGTH_LEVEL = [
    # '16k',
    # '32k',
    # '64k',
    '128k',
    # '256k',
]