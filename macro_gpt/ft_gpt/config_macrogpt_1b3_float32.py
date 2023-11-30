# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/3/5 21:29
# @author  : Mo
# @function: config of macrogpt


# optimized for RTX 4090. for larger GPUs, increase some of these?
# MICRO_BATCH_SIZE = 4  # default=4  # this could actually be 5 but i like powers of 2
# MICRO_BATCH_SIZE = 16  # default=4  # this could actually be 5 but i like powers of 2
MICRO_BATCH_SIZE = 4  # default=4  # this could actually be 5 but i like powers of 2
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 3e-4  # default=3e-4  # the Karpathy constant
EPOCHS = 1  # default=3  # we don't always need 3 tbh
# LORA_DROPOUT = 0.1
# LORA_ALPHA = 32
# LORA_R = 32
LORA_DROPOUT = 0.05
LORA_ALPHA = 16
LORA_R = 8
SAVE_STEPS = 384
VAL_SET_SIZE = 0
MAX_LENGTH_Q = 1024 - 2  # default=128 - 2
MAX_LENGTH_A = 1024 - 2  # default=128 - 2
MAX_LENGTH_QA = MAX_LENGTH_Q + MAX_LENGTH_A + 4
TARGET_MODULES = ["query_key_value"]

PATH_MODEL_PRETRAIN = ""
REPO_ID = "Macropodus/macrogpt-tokenizer"
PATH_MODEL_PRETRAIN = PATH_MODEL_PRETRAIN if PATH_MODEL_PRETRAIN else REPO_ID
DATA_PATH = "../datasets/tigerbot-train-00001-of-00097.json"
MODEL_SAVE_DIR = "model_macrogpt_1b3_float32"
PATH_TOKENIZER_PRETRAIN = REPO_ID or "./macrogpt.model"
PATH_MODEL_CONFIG = "config_macrogpt_1b3_float32.json" or MODEL_SAVE_DIR

IS_PARALLELIZABLE = False
MODEL_PARALLEL = False
USE_CACHE = False
CUDA_VISIBLE_DEVICES = "0"
USE_TORCH = "1"
CPU_NUMS = "9"
USE_CUDA = False if CUDA_VISIBLE_DEVICES == "-1" else True

