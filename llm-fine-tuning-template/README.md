# Qwen3 DPO Fine-tuning 项目

这个项目使用TRL (Transformer Reinforcement Learning) 库对Qwen3-4B模型进行DPO (Direct Preference Optimization) fine-tuning。

## 项目结构

```
├── requirements.txt          # 项目依赖
├── README.md                # 项目说明
├── dpo_train.py            # DPO训练主脚本
├── data_utils.py           # 数据处理工具
├── model_utils.py          # 模型相关工具
├── config.py               # 配置文件
├── train_config.yaml       # 训练配置
└── data/                   # 数据目录
    └── dpo_dataset.json    # DPO训练数据
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. 准备数据：将DPO格式的数据放在 `data/dpo_dataset.json`
2. 配置训练参数：修改 `train_config.yaml`
3. 开始训练：`python dpo_train.py`

## DPO数据格式

数据应该是以下格式的JSON文件：

```json
[
  {
    "prompt": "用户的问题或指令",
    "chosen": "更好的回答",
    "rejected": "较差的回答"
  }
]
```

## 模型信息

- 基础模型：Qwen3-4B (https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f)
- 训练方法：DPO (Direct Preference Optimization)
- 训练框架：TRL (Transformer Reinforcement Learning)
