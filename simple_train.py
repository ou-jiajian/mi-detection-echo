#!/usr/bin/env python3
"""简化版训练脚本，无wandb依赖"""
import os
import sys
import torch
import tqdm
import numpy as np
from easydict import EasyDict
from torch.utils.data import DataLoader
from datasets.segmentation_dataset import LVWallDataset
from segmentation.segmentation_utils import *
from utils.videos import AverageMeter

def train_one_epoch(epoch, model, optimizer, train_loader, device):
    model.train()
    pbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
    loss_all, iou_all = AverageMeter(), AverageMeter()

    for step, batch in pbar:    
        img, ytrue = batch[0].to(device).float(), batch[1].to(device).float()
        ypred = torch.sigmoid(model(img))
        
        loss = get_loss(ypred, ytrue, batch)
        iou = get_iou(ypred, ytrue)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_all.update(loss.item(), img.shape[0])
        iou_all.update(iou.item(), img.shape[0])
        
        pbar.set_description(f"train epoch {epoch + 1} batch {step + 1} / {len(pbar)} train-loss {loss_all.avg:.4f} train-iou {iou_all.avg:.4f}") 
    
    return loss_all, iou_all

def valid_one_epoch(epoch, model, val_loader, device):
    model.eval()
    pbar = tqdm.tqdm(enumerate(val_loader), total=len(val_loader))
    loss_all, iou_all = AverageMeter(), AverageMeter()
    
    for step, batch in pbar:
        img, ytrue = batch[0].to(device).float(), batch[1].to(device).float()
        
        with torch.no_grad():
            ypred = torch.sigmoid(model(img))
            loss = get_loss(ypred, ytrue, batch)
            iou = get_iou(ypred, ytrue)
                
        loss_all.update(loss.item(), img.shape[0])
        iou_all.update(iou.item(), img.shape[0])
        
        pbar.set_description(f"test epoch {epoch + 1} batch {step + 1} / {len(pbar)} test-loss {loss_all.avg:.4f} test-iou {iou_all.avg:.4f}")
    
    return loss_all, iou_all

def main():
    device = torch.device('cpu')  # 使用CPU
    
    config = EasyDict(dict(
        architecture='Unet',
        encoder='resnet18',
        epoch=2,  # 训练2个epoch
        fold=0,
        batch_size=4,
        img_shape=(224, 224),
    ))
    
    print(f"开始训练，配置: {config}")
    
    # 数据集
    train_set = LVWallDataset(split='train', id_fold=config.fold, img_size=config.img_shape)
    val_set = LVWallDataset(split='test', id_fold=config.fold, img_size=config.img_shape)
    
    train_loader = DataLoader(train_set, config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, config.batch_size, shuffle=False)
    
    print(f"训练集: {len(train_set)} 样本, 测试集: {len(val_set)} 样本")
    
    # 模型
    model, optimizer = get_model_and_optim(config, device=device)
    
    # 模型保存目录
    save_dir = os.path.join(os.getcwd(), 'models', 'simple_train')
    os.makedirs(save_dir, exist_ok=True)
    
    best_iou = 0
    
    for epoch in range(config.epoch):
        print(f"\n=== Epoch {epoch+1}/{config.epoch} ===")
        
        train_loss, train_iou = train_one_epoch(epoch, model, optimizer, train_loader, device)
        test_loss, test_iou = valid_one_epoch(epoch, model, val_loader, device)
        
        # 保存模型
        model_path = os.path.join(save_dir, f'epoch_{epoch+1}_train_{train_iou.avg:.4f}_test_{test_iou.avg:.4f}.pth')
        torch.save(model.state_dict(), model_path)
        
        print(f"Epoch {epoch+1} 结果:")
        print(f"  训练: loss={train_loss.avg:.4f}, iou={train_iou.avg:.4f}")
        print(f"  测试: loss={test_loss.avg:.4f}, iou={test_iou.avg:.4f}")
        print(f"  模型已保存: {model_path}")
        
        if test_iou.avg > best_iou:
            best_iou = test_iou.avg
            best_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_path)
            print(f"  🏆 新的最佳模型! IoU={best_iou:.4f}")
    
    print(f"\n✅ 训练完成! 最佳测试IoU: {best_iou:.4f}")

if __name__ == '__main__':
    main()
