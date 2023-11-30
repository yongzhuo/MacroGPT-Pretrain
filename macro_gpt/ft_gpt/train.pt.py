# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2023/3/5 21:04
# @author  : Mo
# @function: macro-gpt


import random
import copy
import sys
import os
path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(path_root)
from macro_gpt.ft_gpt.config_macrogpt_1b3_float32 import CUDA_VISIBLE_DEVICES, USE_TORCH, CPU_NUMS  # from config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:3072"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
os.environ["USE_TORCH"] = USE_TORCH
os.environ["OMP_NUM_THREADS"] = CPU_NUMS  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = CPU_NUMS  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = CPU_NUMS  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = CPU_NUMS  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = CPU_NUMS  # export NUMEXPR_NUM_THREADS=1

from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from peft import (get_peft_model_state_dict, get_peft_model, LoraConfig)
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_utils import unwrap_model
from tensorboardX import SummaryWriter
from datasets import load_dataset
import bitsandbytes as bnb
import torch.nn as nn
import transformers
import torch

from macro_gpt.models.llama.modeling_llama import LlamaForCausalLM as LLMForCausalLM
from macro_gpt.models.llama.tokenization_llama import LlamaTokenizer as LLMTokenizer
from macro_gpt.models.llama.modeling_llama import LlamaConfig as LLMConfig
from macro_gpt.ft_gpt.config_macrogpt_1b3_float32 import PATH_MODEL_PRETRAIN, DATA_PATH, MODEL_SAVE_DIR, REPO_ID
from macro_gpt.ft_gpt.config_macrogpt_1b3_float32 import MICRO_BATCH_SIZE, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS
from macro_gpt.ft_gpt.config_macrogpt_1b3_float32 import LEARNING_RATE, EPOCHS, SAVE_STEPS, VAL_SET_SIZE, TARGET_MODULES
from macro_gpt.ft_gpt.config_macrogpt_1b3_float32 import IS_PARALLELIZABLE, MODEL_PARALLEL, USE_CACHE
from macro_gpt.ft_gpt.config_macrogpt_1b3_float32 import MAX_LENGTH_Q, MAX_LENGTH_A, MAX_LENGTH_QA
from macro_gpt.ft_gpt.config_macrogpt_1b3_float32 import LORA_DROPOUT, LORA_ALPHA, LORA_R
from macro_gpt.ft_gpt.config_macrogpt_1b3_float32 import PATH_MODEL_CONFIG, PATH_TOKENIZER_PRETRAIN



def save_model_state(model, config=None, model_save_dir="./", model_name="adapter_model.bin"):
    """  仅保存 有梯度 的 模型参数(推荐使用)  """
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    # save config
    if config:
        config.save_pretrained(model_save_dir)
        # config.to_dict()
    # save model
    path_model = os.path.join(model_save_dir, model_name)
    # grad_params_dict = {k: v.to("cpu") for k, v in model.named_parameters()
    #                     if v.requires_grad == True}
    grad_params_dict = {k: v.to("cpu") for k, v in model.named_parameters()}
    torch.save(grad_params_dict, path_model)
    print_rank_0("******model_save_path is {}******".format(path_model))
def print_rank_0_named_parameters(model, use_print_rank_0_data=False):
    """   打印模型训练参数/数据类型信息   """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        if use_print_rank_0_data:
            print_rank_0((name, param.data.dtype, param.requires_grad, param.data))
        else:
            print_rank_0((name, param.data.dtype, param.requires_grad))
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print_rank_0(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
def prepare_model_for_half_training(model, output_embedding_layer_name="lm_head",
        use_gradient_checkpointing=True, layer_norm_names=["layer_norm"]):
    r"""
    This method wrapps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    #  不要使用 model.half(), 这样会先截取精度再训练了, 最初data就要保持half
    for name, param in model.named_parameters():
        # freeze base model's layers
        # cast layer norm in fp32 for stability for 8bit models
        if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
            param.data = param.data.to(torch.float32)
        elif output_embedding_layer_name in name:  # lm_head也需要是tf.float32(最后一层)
            param.data = param.data.to(torch.float32)
        else:
            param.data = param.data.to(torch.half)

    if use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()
    return model

def generate_prompt(data_point, is_logger=False):
    # sorry about the formatting disaster gotta move fast
    # text_1 = f"指令：\n{data_point.get('instruction', '')}\n问：\n{data_point.get('input', '')}\n答：\n" \
    #     if data_point.get('input', '') else f"指令：\n{data_point.get('instruction', '')}\n答：\n"
    # text_2 = f"{data_point.get('output', '')}"
    text_a = data_point.get("a", "")
    prompt_str_1 = text_a
    # end with gMASK, <sop>
    x = tokenizer.encode(prompt_str_1)
    if len(x) > MAX_LENGTH_QA - 2:
        x = x[:MAX_LENGTH_QA - 2]
    if not x:
        x = [ID_PAD, ID_EOS]
    if x and x[-1] != ID_EOS:
        x += [ID_EOS]
    out = {"input_ids": x, "labels": []}
    if is_logger:
        print_rank_0(prompt_str_1)
        print_rank_0(out)
    return out


def data_collator(batch):
    def get_position_ids(seq, bos_token_id):
        seq_length = len(seq)
        position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)
        return position_ids
    def get_masks(seq, special_ids=IDS_ORG):
        """  padding-mask  """
        # mask until ID_SOP
        attention_mask = torch.ones((1, len(seq), len(seq)))
        attention_mask.tril_()
        # ### 如果 padding-right, 也mask掉
        # for idx, s in enumerate(seq):
        #     if s in special_ids:
        #         attention_mask[..., idx] = 1
        attention_mask = (attention_mask < 0.5).bool()
        return attention_mask

    len_max_batch = [len(batch[i].get("input_ids")) + len(batch[i].get("labels")) + 1
                     for i in range(len(batch))]
    len_max_batch = min(MAX_LENGTH_QA, max(len_max_batch))
    batch_attention_mask = []
    batch_position_ids = []
    batch_input_ids = []
    batch_labels = []
    for ba in batch:
        x, y = ba.get("input_ids"), ba.get("labels")
        len_padding = len_max_batch - len(x) - len(y)
        if tokenizer.padding_side and tokenizer.padding_side == "left":
            labels = [-100] * len_padding + x + y
            input_ids = [ID_PAD] * (len_padding) + x + y
        else:
            labels = x + y + [-100] * len_padding
            input_ids = x + y + [ID_PAD] * (len_padding)
        tensor_position_ids = get_position_ids(input_ids, bos_token_id=ID_SOP)
        tensor_attention_mask = get_masks(input_ids, special_ids=IDS_ORG)
        tensor_input_ids = torch.tensor(input_ids, dtype=torch.long)
        tensor_labels = torch.tensor(labels, dtype=torch.long)
        batch_attention_mask.append(tensor_attention_mask)
        batch_position_ids.append(tensor_position_ids)
        batch_input_ids.append(tensor_input_ids)
        batch_labels.append(tensor_labels)
    # print_rank_0(batch_attention_mask)
    batch_attention_mask = torch.stack(batch_attention_mask)
    batch_position_ids = torch.stack(batch_position_ids)
    batch_input_ids = torch.stack(batch_input_ids)
    batch_labels = torch.stack(batch_labels)
    input_dict = {
                  # "full_attention_mask": copy.deepcopy(batch_attention_mask),
                  # "attention_mask": batch_attention_mask,
                  # "position_ids": batch_position_ids,
                  "input_ids": batch_input_ids,
                  "labels": batch_labels,
                  }
    # print_rank_0(input_dict)
    return input_dict
def dfs_file(path_dir):
    """
        递归获取某个目录下的所有文件(所有层, 包括子目录)
    Args:
        path_dir[String]:, path of dir, eg. "/home/data"
    Returns:
        data[List]: data of input, eg. ["2020_01_08.txt"]
    """
    path_files = []
    for root, dirs, files in os.walk(path_dir):  # 分别代表根目录、文件夹、文件
        for file in files:  # 遍历文件
            file_path = os.path.join(root, file)  # 获取文件绝对路径
            path_files.append(file_path)  # 将文件路径添加进列表
    files = list(set(path_files))
    files.sort()  # the same list
    return files
def print_rank_0(*args):
    """   只打印 0 号GPU的   """
    # if torch.distributed.get_rank() == 0:  # 一般用0，当然，可以选任意的rank保存。
    #     print(*args)
    print(*args)
def local_rank_is_0():
    """   判断是哪台机子的  """
    # flag = False
    # if torch.distributed.get_rank() == 0:
    #     flag = True
    # return flag
    return True


# import torch.distributed as dist
# dist.init_process_group(backend='nccl')

# torch.distributed.init_process_group()
tokenizer = LLMTokenizer.from_pretrained(PATH_TOKENIZER_PRETRAIN)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "left"  # Allow batched inference
tokenizer.padding_side = "right"  # Allow batched inference
# ID_gMASK = 64790
# ID_BOS = 64792
# ID_EOS = 64793
# ID_MASK = 64789
# ID_PAD = 2
ID_EOP = 2
ID_SOP = 1
ID_BOS = 1
ID_EOS = 2
ID_PAD = 0

IDS_ORG = [ID_PAD]
# { "<|endoftext|>": 50256,
#   "### End": 50257,
#   "### Instruction:": 50258,
#   "### Response:\n": 50259
# }

# model = GPT2LMHeadModel.from_pretrained(PATH_MODEL_PRETRAIN)
llm_config = LLMConfig.from_json_file(PATH_MODEL_CONFIG)
model = LLMForCausalLM(llm_config)
model.init_weights()
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.is_parallelizable = IS_PARALLELIZABLE
model.model_parallel = MODEL_PARALLEL
model.config.use_cache = USE_CACHE
# model.clip_grad_norm_ = 1.0
# model = model.half().cuda()
## norm, lm_head层为fp32
# prepare_model_for_half_training(model, output_embedding_layer_name="lm_head",
#         use_gradient_checkpointing=True, layer_norm_names=["post_attention_layernorm",
#                                                            "input_layernorm",
#                                                            "norm",
#                                                            ])
model = model.cuda()
print_rank_0_named_parameters(model)

tensorboardx_witer = SummaryWriter(logdir=MODEL_SAVE_DIR)


# files = dfs_file(DATA_PATH)
# files = [files for file in files if "data_merge.0" in file or "data_merge.1" in file]
### 只有一个train的情况
# data = load_dataset("json", data_files={"train": files})
data = load_dataset("json", data_files=DATA_PATH)
# data = load_dataset("json", data_dir=DATA_PATH)


# train_val = data["train"].train_test_split(test_size=min(VAL_SET_SIZE,
#                     int(len(data["train"])/10000)), shuffle=True, seed=42)
# VAL_SET_SIZE = max(min(VAL_SET_SIZE, int(len(data["train"])/10000)), 1)
# generate_prompt(data["train"][0], is_logger=True)
# train_val = data["train"].train_test_split(test_size=VAL_SET_SIZE, shuffle=True, seed=42)
# train_data = train_val["train"].shuffle().map(generate_prompt)
# val_data = train_val["test"].shuffle().map(generate_prompt)

# generate_prompt(data["train"][0], is_logger=True)
# train_val = data["train"].train_test_split(test_size=1024, shuffle=True, seed=42)
# train_data = train_val["test"].shuffle().map(generate_prompt)
# val_data = None

generate_prompt(data["train"][0], is_logger=True)
train_data = data["train"].shuffle().map(generate_prompt)
val_data = None


class CustomTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = model(**inputs)  # if contain labels, will calculate loss

        if local_rank_is_0:
            logs = {}
            tr_loss_scalar = self._nested_gather(outputs.loss.detach()).mean().item()
            logs["loss"] = round(tr_loss_scalar, 4)
            logs["lr"] = self.lr_scheduler.get_last_lr()[0]
            step = self.state.global_step
            for k, v in logs.items():
                tensorboardx_witer.add_scalar(k, v, step)
            self.log(logs)

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # if llm_config.torch_dtype == "float16":
        #     loss = loss.half()
        loss = loss.half()
        return (loss, outputs) if return_outputs else loss


trainer = CustomTrainer(
        # data_collator=transformers.DataCollatorForSeq2Seq(
        #                     tokenizer, pad_to_multiple_of=8,
        #                     return_tensors="pt", padding=True
        #                 ),
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=val_data,
        model=model,
        args=transformers.TrainingArguments(
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            per_device_train_batch_size=MICRO_BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            num_train_epochs=EPOCHS,
            max_grad_norm=1.0,
            logging_steps=20,
            # warmup_steps=32,
            # warmup_steps=382,  # 618
            warmup_ratio=0.01,
            # warmup_steps=16,
            evaluation_strategy="no",
            # lr_scheduler_type="constant", #'constant',  # "cosine",
            logging_first_step=False,
            # evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
            # eval_steps=SAVE_STEPS if VAL_SET_SIZE > 0 else None,
            save_strategy="steps",
            save_total_limit=6,
            save_steps=SAVE_STEPS,
            # load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
            # ddp_find_unused_parameters=None,
            gradient_checkpointing=True,
            # group_by_length=True,  # group together samples of roughly the same length in training
            output_dir=MODEL_SAVE_DIR,
            optim="adafactor",  # "adamw_torch",  # "adamw_hf",
            report_to=[],  # ["tensorboard"],  # [], ["wandb"]
            fp16=True,
        )
    )


files = dfs_file(MODEL_SAVE_DIR)
files_name_str = str(files)
flag_checkpoint = True if files and "checkpoint" in files_name_str else False
print_rank_0("flag_checkpoint: {}".format(flag_checkpoint))
trainer.train(resume_from_checkpoint=flag_checkpoint)
old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

save_model_state(model=model, config=llm_config, model_save_dir=MODEL_SAVE_DIR)
print_rank_0_named_parameters(model, use_print_rank_0_data=True)  # 查看LoRA层权重是不是为NAN溢出


# nohup python train.pt.py > tc.train.pt.py.log 2>&1 &
# tail -n 1000  -f tc.train.pt.py.log

