#!/usr/bin/env python3
"""运行原版训练脚本的简单包装器"""
import os
import sys
sys.path.append(os.getcwd())

from easydict import EasyDict
from segmentation.segmentation_trainer import run_n_epochs

def main():
    # 配置参数 - 与原版项目一致
    config = EasyDict(dict(
        architecture='Unet',      # 可选: Unet, UnetPlusPlus, Linknet, FPN, PAN, DeepLabV3
        encoder='resnet18',       # 可选: resnet18, resnet34, mobilenet_v2, efficientnet-b0
        epoch=3,                  # 训练轮数
        fold=0,                   # 交叉验证折数 (0-4)
        batch_size=8,             # 批大小
        img_shape=(224, 224),     # 图像尺寸
    ))
    
    print("=== 使用原版训练脚本 ===")
    print(f"配置: {config}")
    print("这是完整的原版训练流程，包括:")
    print("- 完整的数据加载和预处理")
    print("- 原版的模型架构和训练逻辑") 
    print("- 原版的评估和模型保存逻辑")
    print("- wandb实验跟踪 (离线模式)")
    print("-" * 50)
    
    # 调用原版训练函数
    run_n_epochs(config)
    
    print("\n✅ 原版训练流程完成!")

if __name__ == '__main__':
    main()
