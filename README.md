## 仓库信息

本项目的 GitHub 仓库： [ou-jiajian/mi-detection-echo](https://github.com/ou-jiajian/mi-detection-echo)

## 环境配置（Conda）
1. 在 `environment.yml` 中完善依赖库
2. 创建环境（前缀在当前目录的 `./env`）
```bash
conda env create --prefix ./env --file environment.yml
```
3. 激活环境
```bash
conda activate ./env
```
4. 更新依赖
```bash
conda env update --prefix ./env --file environment.yml --prune
```

## 基于超声心动图的心肌梗死检测

![左心室六分区示意图](lv-6segments.png)
- (A) HCM-QU 数据集中的两帧（收缩末期与舒张末期）左心室（LV）心肌分割示意，分别对应 MI 与非 MI 个案。
- (B) LV 壁被划分为 6 个分区用于 MI 迹象检测。标记 “L” 表示从左下角至心尖的长度，标记 “R” 表示从右下角至心尖的长度。

## 框架总览

![框架总览](overview-framework.png)
基于两阶段的 MI 检测框架：
- Phase 01：编码器-解码器结构进行分割与表征（蓝色为卷积层，灰色为反卷积层，蓝色箭头为跳连）。
- Phase 02：利用心肌位移 𝔻、特征权重 𝕎 与特征集 𝔼 的集成策略进行 MI 判别。

## 引用
如使用本仓库，请引用如下论文：
```
@article{nguyen2023ensemble,
  title={Ensemble learning of myocardial displacements for myocardial infarction detection in echocardiography},
  author={Nguyen, Tuan and Nguyen, Phi and Tran, Dai and Pham, Hung and Nguyen, Quang and Le, Thanh and Van, Hanh and Do, Bach and Tran, Phuong and Le, Vinh and others},
  journal={Frontiers in Cardiovascular Medicine},
  volume={10},
  year={2023},
  publisher={Frontiers Media SA}
}
```

## 说明
- 本中文 `README.md` 为项目主要说明文档。
- 英文副本请见 `README.en.md`。