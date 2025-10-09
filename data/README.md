# Data

We adopt the same dataset organization scheme as [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main/data).  
You can add your own customized datasets by updating [dataset_info.json](./dataset_info.json).  
Multiple datasets can be specified in the training script by:

- Using `","` within `--dataset` to concatenate multiple datasets, e.g., `chatqa2,my_dataset`
- Controlling sampling probabilities with `--interleave_probs`, e.g., `--interleave_probs 0.1,0.5`.  