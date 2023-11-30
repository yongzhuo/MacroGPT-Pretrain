# macrogpt-prertrain
大模型全量预训练(1b3), 多卡deepspeed/单卡adafactor

## 踩坑
```python
1. 数据类型fp16不太行, 很容易就Nan了, 最好是fp32, tf32,
2. 单卡如果显存不够, 可以用优化器'adafactor',
3. 如果数据量很大, 加载时间特别长(默认设置稍微大一点数据就得加载好几个小时), 可以分批次训练,
```

## 环境配置
```shell
transformers>=4.31.0
torch>=1.10.1
rouge==1.0.1
nltk==3.6.6
peft>=0.2.0
numpy
tqdm
```

## 预训练
```shell
地址: macro_gpt/ft_gpt

配置: macro_gpt/ft_gpt/config.llama_1b3_float32.json
单卡第一次训练: python train.pt.py
单卡继续训练: python train.pt.add.py
多卡训练: deepspeed --num_gpus=2 train.pt.speed.py --deepspeed ds.json
```

## 预训练日志(TigerBot-en)
图为tigerbot-en-00001-of-00097.json的预训练日志, loss收敛到3左右

![macro_gpt/macro_gpt_loss.png](macro_gpt/macro_gpt_loss.png)

图为baidu百科数据集(第一个60w,此外还有10%领域专业数据)的预训练日志, loss收敛到3左右
![macro_gpt/macro_gpt_zh_loss.png](macro_gpt/macro_gpt_zh_loss.png)


## 预测日志
一问一答还行, 1b3的大模型上下文能力确实比较弱

![macro_gpt/macro_gpt_pt.png](macro_gpt/macro_gpt_pt.png)


## 数据集-中文
 - [https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)
 - [https://github.com/TigerResearch/TigerBot](https://github.com/TigerResearch/TigerBot)

## 参考/感谢
 - [https://github.com/tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)
 - [https://github.com/huggingface/peft](https://github.com/huggingface/peft)
 - [trl](https://github.com/lvwerra/trl)

## 免责申明
本项目相关资源仅供学术研究之用，使用涉及第三方代码的部分时，请严格遵循相应的开源协议。模型生成的内容受模型计算、随机性和量化精度损失等因素影响，本项目不对其准确性作出保证。对于模型输出的任何内容，本项目不承担任何法律责任，亦不对因使用相关资源和输出结果而可能产生的任何损失承担责任。

