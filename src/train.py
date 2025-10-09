# Copyright 2025 the LlamaFactory team.
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under Apache-2.0, with the full license text
# available at https://github.com/hiyouga/LLaMA-Factory/blob/main/LICENSE.
#
# This modified file is released under the same license.

from llamafactory.train.tuner import run_exp


def main():
    # Customized models
    from ahn.transformer.qwen2_ahn import register_customized_qwen2
    from ahn.transformer.qwen3_ahn import register_customized_qwen3
    
    register_customized_qwen2()
    register_customized_qwen3()

    run_exp()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    run_exp()


if __name__ == "__main__":
    main()
