@echo off
chcp 65001 >nul

echo ==========================================
echo Qwen3 DPO Fine-tuning 项目
echo ==========================================

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python
    pause
    exit /b 1
)

REM 检查CUDA
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo 警告: 未检测到NVIDIA GPU，将使用CPU训练（速度会很慢）
) else (
    echo 检测到NVIDIA GPU:
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
)

REM 创建必要的目录
echo 创建项目目录...
if not exist "data" mkdir data
if not exist "outputs" mkdir outputs
if not exist "logs" mkdir logs

REM 安装依赖
echo 安装项目依赖...
pip install -r requirements.txt

REM 检查依赖安装
echo 检查依赖安装...
python -c "import torch; import transformers; import trl; import datasets; import peft; print(f'PyTorch版本: {torch.__version__}'); print(f'Transformers版本: {transformers.__version__}'); print(f'TRL版本: {trl.__version__}'); print(f'Datasets版本: {datasets.__version__}'); print(f'PEFT版本: {peft.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}' if torch.cuda.is_available() else 'GPU数量: 0')"

REM 创建示例数据
echo 准备训练数据...
python data_utils.py

REM 开始训练
echo 开始DPO训练...
echo 注意: 训练过程可能需要较长时间，请耐心等待...
echo 训练日志将保存到 dpo_training.log
echo 模型将保存到 outputs/dpo_qwen3_4b/

python dpo_train.py

echo ==========================================
echo 训练完成！
echo ==========================================
echo 训练后的模型保存在: outputs/dpo_qwen3_4b/
echo 训练日志保存在: dpo_training.log
echo.
echo 要测试模型，请运行: python test_model.py
pause
