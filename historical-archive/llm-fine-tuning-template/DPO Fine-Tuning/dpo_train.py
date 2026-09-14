#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) 训练脚本
使用TRL库对Qwen3-4B模型进行fine-tuning
"""

import os
import sys
import logging
import yaml
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer
from trl import DPOTrainer
from trl.trainer import ConstantLengthDataset

from data_utils import load_dpo_dataset, create_sample_dpo_data
from model_utils import (
    load_model_and_tokenizer,
    create_peft_config,
    apply_peft_to_model,
    create_training_arguments,
    save_model_and_tokenizer
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dpo_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "train_config.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def prepare_dataset(config: dict):
    """准备训练数据集"""
    data_config = config['data']
    train_file = data_config['train_file']
    
    # 如果数据文件不存在，创建示例数据
    if not os.path.exists(train_file):
        logger.info("数据文件不存在，创建示例数据...")
        create_sample_dpo_data(train_file)
    
    # 加载数据集
    dataset = load_dpo_dataset(train_file)
    
    # 分割训练集和验证集
    test_size = data_config.get('test_size', 0.1)
    if test_size > 0:
        dataset = dataset.train_test_split(test_size=test_size)
        train_dataset = dataset['train']
        eval_dataset = dataset['test']
        logger.info(f"数据集分割完成: 训练集 {len(train_dataset)} 条, 验证集 {len(eval_dataset)} 条")
    else:
        train_dataset = dataset
        eval_dataset = dataset
        logger.info(f"使用全部数据作为训练集: {len(train_dataset)} 条")
    
    return train_dataset, eval_dataset

def create_dpo_trainer(
    model,
    tokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    config: dict
):
    """创建DPO训练器"""
    
    # 创建训练参数
    training_config = config['training']
    output_config = config['output']
    
    training_args = create_training_arguments(
        output_dir=output_config['output_dir'],
        num_train_epochs=training_config['num_train_epochs'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        learning_rate=training_config['learning_rate'],
        warmup_steps=training_config['warmup_steps'],
        logging_steps=training_config['logging_steps'],
        save_steps=training_config['save_steps'],
        eval_steps=training_config['eval_steps'],
        save_total_limit=training_config['save_total_limit'],
        load_best_model_at_end=training_config['load_best_model_at_end'],
        metric_for_best_model=training_config['metric_for_best_model'],
        greater_is_better=training_config['greater_is_better'],
        logging_dir=output_config['logging_dir']
    )
    
    # 创建DPO训练器
    dpo_config = config['dpo']
    dpo_trainer = DPOTrainer(
        model=model,
        args=training_args,
        beta=dpo_config['beta'],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_prompt_length=dpo_config['max_prompt_length'],
        max_length=dpo_config['max_length'],
        max_target_length=dpo_config['max_target_length'],
        peft_config=None,  # 已经在模型上应用了PEFT
        is_encoder_decoder=False,
        disable_dropout=False,
        generate_during_eval=False,
        compute_metrics=None,
        precompute_ref_log_probs=False,
        ref_model=None,
        use_dpo_data_collator=True,
        data_collator_kwargs={},
    )
    
    logger.info("DPO训练器创建完成")
    return dpo_trainer

def main():
    """主函数"""
    logger.info("开始DPO训练...")
    
    # 加载配置
    config = load_config()
    logger.info("配置文件加载完成")
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        logger.info(f"CUDA可用，使用GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        logger.warning("CUDA不可用，将使用CPU训练（速度会很慢）")
    
    # 准备数据集
    train_dataset, eval_dataset = prepare_dataset(config)
    
    # 加载模型和分词器
    model_config = config['model']
    hardware_config = config['hardware']
    
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_config['base_model'],
        use_4bit=hardware_config['use_4bit'],
        use_8bit=hardware_config['use_8bit'],
        bf16=hardware_config['bf16'],
        device_map=hardware_config['device_map']
    )
    
    # 应用PEFT配置
    if model_config['use_peft']:
        peft_config = create_peft_config(
            lora_r=model_config['lora_r'],
            lora_alpha=model_config['lora_alpha'],
            lora_dropout=model_config['lora_dropout'],
            target_modules=model_config['target_modules']
        )
        model = apply_peft_to_model(model, peft_config)
    
    # 创建DPO训练器
    dpo_trainer = create_dpo_trainer(
        model, tokenizer, train_dataset, eval_dataset, config
    )
    
    # 开始训练
    logger.info("开始训练...")
    train_result = dpo_trainer.train()
    
    # 保存训练结果
    output_dir = config['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存模型和分词器
    save_model_and_tokenizer(model, tokenizer, output_dir)
    
    # 保存训练结果
    dpo_trainer.save_model()
    dpo_trainer.save_state()
    
    # 保存训练指标
    metrics_file = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_file, 'w') as f:
        import json
        json.dump(train_result.metrics, f, indent=2)
    
    logger.info(f"训练完成！模型已保存到: {output_dir}")
    logger.info(f"训练指标已保存到: {metrics_file}")
    
    # 打印最终指标
    logger.info("最终训练指标:")
    for key, value in train_result.metrics.items():
        logger.info(f"  {key}: {value}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}", exc_info=True)
        sys.exit(1)
