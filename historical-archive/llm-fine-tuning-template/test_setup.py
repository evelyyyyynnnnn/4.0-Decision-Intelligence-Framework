#!/usr/bin/env python3
"""
测试项目设置和依赖
"""

import os
# 设置环境变量抑制TensorFlow AVX警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import importlib

def test_imports():
    """测试必要的包是否能正常导入"""
    required_packages = [
        'torch',
        'transformers', 
        'trl',
        'datasets',
        'peft',
        'accelerate',
        'bitsandbytes'
    ]
    
    print("🔍 测试包导入...")
    failed_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n❌ 以下包导入失败: {', '.join(failed_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ 所有必要的包都能正常导入")
        return True

def test_cuda():
    """测试CUDA可用性"""
    print("\n🔍 测试CUDA...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA可用")
            print(f"   GPU数量: {torch.cuda.device_count()}")
            print(f"   当前GPU: {torch.cuda.get_device_name()}")
            print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            return True
        else:
            print("⚠️  CUDA不可用，将使用CPU训练（速度会很慢）")
            return False
    except Exception as e:
        print(f"❌ CUDA测试失败: {e}")
        return False

def test_data():
    """测试数据文件"""
    print("\n🔍 测试数据文件...")
    
    import os
    data_file = "data/dpo_dataset.json"
    
    if os.path.exists(data_file):
        print(f"✅ 数据文件存在: {data_file}")
        
        # 检查文件大小
        file_size = os.path.getsize(data_file)
        print(f"   文件大小: {file_size} bytes")
        
        # 尝试加载数据
        try:
            from data_utils import load_dpo_dataset
            dataset = load_dpo_dataset(data_file)
            print(f"   数据条数: {len(dataset)}")
            print("✅ 数据加载成功")
            return True
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False
    else:
        print(f"❌ 数据文件不存在: {data_file}")
        return False

def test_config():
    """测试配置文件"""
    print("\n🔍 测试配置文件...")
    
    import os
    config_file = "train_config.yaml"
    
    if os.path.exists(config_file):
        print(f"✅ 配置文件存在: {config_file}")
        
        try:
            import yaml
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            print("✅ 配置文件格式正确")
            print(f"   基础模型: {config['model']['base_model']}")
            print(f"   训练轮数: {config['training']['num_train_epochs']}")
            print(f"   DPO beta: {config['dpo']['beta']}")
            return True
        except Exception as e:
            print(f"❌ 配置文件解析失败: {e}")
            return False
    else:
        print(f"❌ 配置文件不存在: {config_file}")
        return False

def main():
    """主函数"""
    print("🚀 Qwen3 DPO Fine-tuning 项目设置测试")
    print("=" * 50)
    
    tests = [
        ("包导入测试", test_imports),
        ("CUDA测试", test_cuda),
        ("数据文件测试", test_data),
        ("配置文件测试", test_config)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name}出现异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！项目设置完成，可以开始训练了。")
        print("\n🚀 开始训练:")
        print("   Linux/Mac: ./run_training.sh")
        print("   Windows: run_training.bat")
        print("   手动: python dpo_train.py")
    else:
        print("⚠️  部分测试失败，请检查配置后重试。")
        print("\n💡 建议:")
        print("   1. 运行: pip install -r requirements.txt")
        print("   2. 检查CUDA安装")
        print("   3. 确保网络连接正常（用于下载模型）")

if __name__ == "__main__":
    main()
