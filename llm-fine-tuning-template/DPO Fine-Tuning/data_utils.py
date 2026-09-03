import json
import os
from typing import List, Dict, Any
from datasets import Dataset
import logging

logger = logging.getLogger(__name__)

def load_dpo_dataset(file_path: str) -> Dataset:
    """
    加载DPO格式的数据集
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        Dataset: HuggingFace数据集对象
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 验证数据格式
    for item in data:
        if not all(key in item for key in ['prompt', 'chosen', 'rejected']):
            raise ValueError("数据格式错误，每个样本必须包含 'prompt', 'chosen', 'rejected' 字段")
    
    logger.info(f"成功加载 {len(data)} 条DPO数据")
    return Dataset.from_list(data)

def create_sample_dpo_data(output_path: str = "data/dpo_dataset.json"):
    """
    创建示例DPO数据用于测试
    
    Args:
        output_path: 输出文件路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    sample_data = [
        {
            "prompt": "请解释什么是机器学习？",
            "chosen": "机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习和改进。通过分析大量数据，机器学习算法可以识别模式并做出预测或决策。",
            "rejected": "机器学习就是让机器学习东西。"
        },
        {
            "prompt": "如何提高英语口语？",
            "chosen": "提高英语口语需要多方面的练习：1) 每天练习发音和语调；2) 与母语者交流；3) 观看英语电影和节目；4) 录音并自我纠正；5) 参加英语角或语言交换活动。",
            "rejected": "多听多说就可以了。"
        },
        {
            "prompt": "什么是可持续发展？",
            "chosen": "可持续发展是指在满足当代人需求的同时，不损害后代人满足其需求能力的发展模式。它包括经济、社会和环境三个维度的平衡发展，强调资源的合理利用和环境保护。",
            "rejected": "可持续发展就是发展经济。"
        },
        {
            "prompt": "如何培养良好的阅读习惯？",
            "chosen": "培养良好阅读习惯的方法：1) 设定固定的阅读时间；2) 选择感兴趣的书籍开始；3) 做读书笔记和思考；4) 加入读书俱乐部；5) 循序渐进，从短篇开始；6) 创造安静的阅读环境。",
            "rejected": "多看书就行了。"
        },
        {
            "prompt": "解释一下区块链技术",
            "chosen": "区块链是一种分布式账本技术，具有以下特点：1) 去中心化，不依赖中央机构；2) 不可篡改，数据一旦写入就无法更改；3) 透明性，所有交易记录公开可见；4) 安全性，通过密码学保证数据安全。",
            "rejected": "区块链就是比特币。"
        }
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"示例DPO数据已保存到: {output_path}")
    return sample_data

def format_dpo_data_for_training(dataset: Dataset, tokenizer, max_length: int = 1024) -> Dataset:
    """
    将DPO数据格式化为训练格式
    
    Args:
        dataset: 原始数据集
        tokenizer: 分词器
        max_length: 最大长度
        
    Returns:
        Dataset: 格式化后的数据集
    """
    def format_example(example):
        # 构建prompt + chosen和prompt + rejected的文本
        chosen_text = f"{example['prompt']}\n\n回答：{example['chosen']}"
        rejected_text = f"{example['prompt']}\n\n回答：{example['rejected']}"
        
        # 分词
        chosen_tokens = tokenizer(
            chosen_text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None
        )
        
        rejected_tokens = tokenizer(
            rejected_text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None
        )
        
        # 构建prompt tokens
        prompt_tokens = tokenizer(
            example['prompt'],
            truncation=True,
            max_length=max_length // 2,
            padding=False,
            return_tensors=None
        )
        
        return {
            "input_ids": prompt_tokens["input_ids"],
            "attention_mask": prompt_tokens["attention_mask"],
            "chosen_input_ids": chosen_tokens["input_ids"],
            "chosen_attention_mask": chosen_tokens["attention_mask"],
            "rejected_input_ids": rejected_tokens["input_ids"],
            "rejected_attention_mask": rejected_tokens["attention_mask"],
        }
    
    return dataset.map(format_example, remove_columns=dataset.column_names)

if __name__ == "__main__":
    # 创建示例数据
    create_sample_dpo_data()
