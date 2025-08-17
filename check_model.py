#!/usr/bin/env python3
"""模型验证脚本：检查训练好的模型"""
import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from datasets.segmentation_dataset import LVWallDataset
from segmentation.segmentation_utils import get_model_and_optim
from easydict import EasyDict

def load_model(model_path):
    """加载训练好的模型"""
    config = EasyDict(dict(
        architecture='Unet',
        encoder='resnet18',
        img_shape=(224, 224),
    ))
    
    device = torch.device('cpu')
    model, _ = get_model_and_optim(config, device=device)
    
    # 加载权重
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, device

def test_model(model_path):
    """测试模型性能"""
    print(f"🔍 检查模型: {os.path.basename(model_path)}")
    print(f"📁 路径: {model_path}")
    
    # 检查文件
    if not os.path.exists(model_path):
        print("❌ 模型文件不存在!")
        return
    
    file_size = os.path.getsize(model_path) / (1024*1024)  # MB
    print(f"📊 文件大小: {file_size:.1f} MB")
    
    try:
        # 加载模型
        model, device = load_model(model_path)
        print("✅ 模型加载成功!")
        
        # 统计参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📈 模型参数: {total_params:,} 总参数, {trainable_params:,} 可训练参数")
        
        # 测试推理
        print("\n🧪 测试推理...")
        test_input = torch.randn(1, 3, 224, 224)  # 批大小1，3通道，224x224
        
        with torch.no_grad():
            output = model(test_input)
            print(f"✅ 推理成功! 输入: {test_input.shape}, 输出: {output.shape}")
            print(f"📊 输出范围: [{output.min():.4f}, {output.max():.4f}]")
        
        # 用真实数据测试
        print("\n🖼️  测试真实数据...")
        ds = LVWallDataset(split='test', id_fold=0, img_size=(224,224))
        x, y_true = ds[0]
        
        x_tensor = torch.from_numpy(x).unsqueeze(0).float()  # 添加batch维度
        with torch.no_grad():
            y_pred = torch.sigmoid(model(x_tensor))
            
        print(f"✅ 真实数据推理成功!")
        print(f"📊 预测掩码范围: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
        print(f"📊 真实掩码: {y_true.sum()} 个前景像素")
        print(f"📊 预测掩码: {(y_pred > 0.5).sum()} 个前景像素")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("🤖 心肌梗死检测模型验证")
    print("=" * 60)
    
    # 查找所有模型文件
    model_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.pth'):
                model_files.append(os.path.join(root, file))
    
    if not model_files:
        print("❌ 未找到任何模型文件!")
        return
    
    print(f"📋 找到 {len(model_files)} 个模型文件:")
    for i, path in enumerate(model_files):
        print(f"  {i+1}. {path}")
    
    # 测试最新的模型（按文件名排序）
    latest_model = max([f for f in model_files if 'segment_ckpt_5folds' in f], key=os.path.getmtime)
    
    print(f"\n🎯 测试最新模型:")
    success = test_model(latest_model)
    
    if success:
        print(f"\n🎉 模型验证成功! 模型已就绪，可用于推理。")
        print(f"💡 使用方法: 加载 {os.path.basename(latest_model)} 进行心肌分割推理")
    else:
        print(f"\n❌ 模型验证失败!")

if __name__ == '__main__':
    main()
