#!/bin/bash

# DPO训练启动脚本

# 设置环境变量抑制TensorFlow AVX警告
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0

echo "=========================================="
echo "Qwen3 DPO Fine-tuning 项目"
echo "=========================================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python3"
    exit 1
fi

# 检查CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "检测到NVIDIA GPU:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "警告: 未检测到NVIDIA GPU，将使用CPU训练（速度会很慢）"
fi

# 创建必要的目录
echo "创建项目目录..."
mkdir -p data
mkdir -p outputs
mkdir -p logs

# 安装依赖
echo "安装项目依赖..."
pip install -r requirements.txt

# 检查依赖安装
echo "检查依赖安装..."
python3 -c "
import torch
import transformers
import trl
import datasets
import peft
print(f'PyTorch版本: {torch.__version__}')
print(f'Transformers版本: {transformers.__version__}')
print(f'TRL版本: {trl.__version__}')
print(f'Datasets版本: {datasets.__version__}')
print(f'PEFT版本: {peft.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU数量: {torch.cuda.device_count()}')
    print(f'当前GPU: {torch.cuda.get_device_name()}')
"

# 创建示例数据
echo "准备训练数据..."
python3 data_utils.py

# 开始训练
echo "开始DPO训练..."
echo "注意: 训练过程可能需要较长时间，请耐心等待..."
echo "训练日志将保存到 dpo_training.log"
echo "模型将保存到 outputs/dpo_qwen3_4b/"

python3 dpo_train.py

echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo "训练后的模型保存在: outputs/dpo_qwen3_4b/"
echo "训练日志保存在: dpo_training.log"
echo ""
echo "要测试模型，请运行: python3 test_model.py"
