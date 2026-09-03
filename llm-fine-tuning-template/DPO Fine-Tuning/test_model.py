#!/usr/bin/env python3
"""
测试训练后的DPO模型
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_trained_model(model_path: str, base_model_name: str = None):
    """加载训练后的模型"""
    logger.info(f"正在加载训练后的模型: {model_path}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # 加载基础模型
    if base_model_name:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 加载PEFT权重
        model = PeftModel.from_pretrained(base_model, model_path)
        logger.info("PEFT模型加载完成")
    else:
        # 直接加载完整模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        logger.info("完整模型加载完成")
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt: str, max_length: int = 512):
    """生成回答"""
    # 构建输入
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    # 移动到GPU
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # 生成回答
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 解码回答
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 移除输入部分，只保留生成的回答
    response = response[len(prompt):].strip()
    
    return response

def test_model():
    """测试模型"""
    # 配置
    model_path = "outputs/dpo_qwen3_4b"  # 训练后的模型路径
    base_model_name = "Qwen/Qwen2.5-4B-Instruct"  # 基础模型名称
    
    # 测试问题
    test_prompts = [
        "请解释什么是机器学习？",
        "如何提高英语口语？",
        "什么是可持续发展？",
        "如何培养良好的阅读习惯？",
        "解释一下区块链技术"
    ]
    
    try:
        # 加载模型
        model, tokenizer = load_trained_model(model_path, base_model_name)
        
        logger.info("开始测试模型...")
        
        for i, prompt in enumerate(test_prompts, 1):
            logger.info(f"\n测试 {i}: {prompt}")
            
            try:
                response = generate_response(model, tokenizer, prompt)
                logger.info(f"回答: {response}")
            except Exception as e:
                logger.error(f"生成回答时出错: {e}")
        
        logger.info("\n模型测试完成！")
        
    except Exception as e:
        logger.error(f"加载模型时出错: {e}")
        logger.info("请确保已经完成训练并保存了模型")

if __name__ == "__main__":
    test_model()
