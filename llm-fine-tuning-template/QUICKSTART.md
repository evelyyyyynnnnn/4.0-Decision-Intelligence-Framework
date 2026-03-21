# 快速开始指南

## 🚀 一键启动训练

### Linux/Mac 用户
```bash
./run_training.sh
```

### Windows 用户
```cmd
run_training.bat
```

## 📋 手动步骤

如果你想要手动控制训练过程，可以按照以下步骤：

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 准备数据
```bash
python data_utils.py
```
这会在 `data/` 目录下创建示例DPO数据。

### 3. 开始训练
```bash
python dpo_train.py
```

### 4. 测试模型
```bash
python test_model.py
```

## ⚙️ 配置说明

主要配置在 `train_config.yaml` 文件中：

- **模型配置**: 使用Qwen3-4B模型，启用LoRA微调
- **训练参数**: 3个epoch，学习率5e-5，批次大小2
- **DPO参数**: beta=0.1，最大长度1024
- **硬件配置**: 4bit量化，bf16精度

## 📊 训练监控

- 训练日志: `dpo_training.log`
- 模型输出: `outputs/dpo_qwen3_4b/`
- 训练指标: `outputs/dpo_qwen3_4b/training_metrics.json`

## 🔧 自定义配置

### 修改模型
编辑 `train_config.yaml` 中的 `model.base_model` 字段

### 调整训练参数
修改 `training` 部分的参数

### 更改DPO参数
调整 `dpo` 部分的配置

## 📝 数据格式

DPO数据格式：
```json
[
  {
    "prompt": "用户问题",
    "chosen": "更好的回答",
    "rejected": "较差的回答"
  }
]
```

## ⚠️ 注意事项

1. **GPU要求**: 推荐使用至少16GB显存的GPU
2. **内存要求**: 至少32GB系统内存
3. **训练时间**: 根据数据量和硬件配置，可能需要几小时到几天
4. **存储空间**: 确保有足够的磁盘空间存储模型和日志

## 🆘 常见问题

### Q: 训练过程中出现CUDA内存不足
A: 减小 `per_device_train_batch_size` 或启用梯度检查点

### Q: 模型加载失败
A: 检查网络连接，确保能访问HuggingFace模型仓库

### Q: 训练速度很慢
A: 检查是否启用了GPU，CPU训练会非常慢

## 📚 更多资源

- [TRL DPO文档](https://huggingface.co/docs/trl/en/dpo_trainer)
- [DPO论文](https://arxiv.org/abs/2306.03676)
- [Qwen3模型](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f)
