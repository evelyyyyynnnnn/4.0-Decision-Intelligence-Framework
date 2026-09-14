import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType
import logging

logger = logging.getLogger(__name__)

def load_model_and_tokenizer(
    model_name: str,
    use_4bit: bool = True,
    use_8bit: bool = False,
    bf16: bool = True,
    device_map: str = "auto"
):
    """
    加载模型和分词器
    
    Args:
        model_name: 模型名称
        use_4bit: 是否使用4bit量化
        use_8bit: 是否使用8bit量化
        bf16: 是否使用bf16精度
        device_map: 设备映射
        
    Returns:
        tuple: (model, tokenizer)
    """
    logger.info(f"正在加载模型: {model_name}")
    
    # 配置量化参数
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif use_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left"
    )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
    )
    
    logger.info("模型和分词器加载完成")
    return model, tokenizer

def create_peft_config(
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: list = None
):
    """
    创建PEFT配置
    
    Args:
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        target_modules: 目标模块列表
        
    Returns:
        LoraConfig: LoRA配置
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none"
    )
    
    logger.info(f"PEFT配置创建完成: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    return peft_config

def apply_peft_to_model(model, peft_config):
    """
    将PEFT配置应用到模型
    
    Args:
        model: 基础模型
        peft_config: PEFT配置
        
    Returns:
        model: 应用PEFT后的模型
    """
    logger.info("正在应用PEFT配置到模型...")
    
    # 冻结基础模型参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 应用PEFT
    model = get_peft_model(model, peft_config)
    
    # 打印可训练参数信息
    model.print_trainable_parameters()
    
    logger.info("PEFT配置应用完成")
    return model

def create_training_arguments(
    output_dir: str,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 2,
    per_device_eval_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 5e-5,
    warmup_steps: int = 100,
    logging_steps: int = 10,
    save_steps: int = 500,
    eval_steps: int = 500,
    save_total_limit: int = 3,
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "eval_loss",
    greater_is_better: bool = False,
    logging_dir: str = "logs"
):
    """
    创建训练参数
    
    Args:
        output_dir: 输出目录
        num_train_epochs: 训练轮数
        per_device_train_batch_size: 每个设备的训练批次大小
        per_device_eval_batch_size: 每个设备的评估批次大小
        gradient_accumulation_steps: 梯度累积步数
        learning_rate: 学习率
        warmup_steps: 预热步数
        logging_steps: 日志记录步数
        save_steps: 保存步数
        eval_steps: 评估步数
        save_total_limit: 保存模型总数限制
        load_best_model_at_end: 是否在结束时加载最佳模型
        metric_for_best_model: 最佳模型指标
        greater_is_better: 指标是否越大越好
        logging_dir: 日志目录
        
    Returns:
        TrainingArguments: 训练参数
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        logging_dir=logging_dir,
        logging_strategy="steps",
        evaluation_strategy="steps",
        save_strategy="steps",
        report_to="wandb" if torch.cuda.is_available() else None,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        dataloader_num_workers=4,
        group_by_length=True,
        ddp_find_unused_parameters=False,
    )
    
    logger.info("训练参数创建完成")
    return training_args

def save_model_and_tokenizer(model, tokenizer, output_dir: str):
    """
    保存模型和分词器
    
    Args:
        model: 训练后的模型
        tokenizer: 分词器
        output_dir: 输出目录
    """
    logger.info(f"正在保存模型和分词器到: {output_dir}")
    
    # 保存分词器
    tokenizer.save_pretrained(output_dir)
    
    # 保存模型
    if hasattr(model, 'save_pretrained'):
        model.save_pretrained(output_dir)
    else:
        # 如果是PEFT模型，需要特殊处理
        model.save_pretrained(output_dir)
    
    logger.info("模型和分词器保存完成")
