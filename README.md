# EEG认知障碍分类特征提取

基于 S-Transform 和 CNN 的 EEG 时频特征提取框架，用于认知障碍（AD/FTD/CN）三分类任务。

## 项目概述

本项目实现了针对 ds004504 数据集的综合特征提取方案，包括：

1. **S-Transform 时频特征** - 基于 Stockwell 变换提取时频域统计特征
2. **CNN 时频特征** - 使用卷积神经网络从时频图中提取深度特征
3. **传统频域特征** - 包括相对频带功率(RBP)、频带比值、α峰值特征

### 方法参考

S-Transform 和 CNN 特征提取方法参考论文：
> Xie et al. (2025). "Attention-augmented stockwell transform and convolutional neural network framework for electroencephalogram-based multi-class classification of Frontotemporal Dementia"

## 数据集信息

- **数据集**: ds004504 (OpenNeuro)
- **受试者**: 88人 (36 AD, 23 FTD, 29 CN)
- **采样率**: 500 Hz
- **电极**: 19通道 (10-20系统)
- **格式**: EEGLAB (.set)

## 特征类型

### 1. 传统特征
| 特征类型 | 描述 |
|---------|------|
| 相对频带功率(RBP) | δ, θ, α, β 频带的相对功率 |
| 频带比值 | θ/α, θ/β, (δ+θ)/(α+β) |
| α峰值特征 | 峰值频率、峰值功率、带宽、中心频率 |

### 2. S-Transform特征
- 时频图统计特征（均值、标准差、最大值等）
- 各频带的时频特征
- 频率质心和扩展
- 时间熵特征

### 3. CNN特征
- 基于3层CNN块提取的深度特征
- 包含注意力机制加权
- 输出维度可配置（默认128维）

## 脑区划分

| 脑区 | 电极 |
|------|------|
| 前额叶 (Frontal) | Fp1, Fp2, F7, F3, Fz, F4, F8 |
| 颞叶 (Temporal) | T3, T4, T5, T6 |
| 顶叶 (Parietal) | P3, Pz, P4 |
| 枕叶 (Occipital) | O1, O2 |

## 频带定义

| 频带 | 范围 (Hz) |
|------|----------|
| δ (delta) | 0.5 - 4 |
| θ (theta) | 4 - 8 |
| α (alpha) | 8 - 13 |
| β (beta) | 13 - 30 |

## 安装

```bash
# 创建虚拟环境（推荐）
conda create -n eeg_feature python=3.9
conda activate eeg_feature

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 基本使用

```bash
# 完整特征提取
python main.py --dataset_path ../dataset/ds004504

# 测试模式（快速验证）
python main.py --dataset_path ../dataset/ds004504 --test_mode

# 不使用CNN特征
python main.py --dataset_path ../dataset/ds004504 --no_cnn

# 使用GPU加速
python main.py --dataset_path ../dataset/ds004504 --device cuda
```

### 参数说明

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--dataset_path` | `../dataset/ds004504` | 数据集路径 |
| `--output_dir` | `./cache` | 输出目录 |
| `--sample_rate` | `500.0` | 采样率 (Hz) |
| `--window_size` | `4.0` | 窗口大小 (秒) |
| `--overlap` | `0.5` | 重叠率 |
| `--use_cnn` | `True` | 使用CNN提取特征 |
| `--no_cnn` | `False` | 不使用CNN |
| `--cnn_feature_dim` | `128` | CNN特征维度 |
| `--device` | `cpu` | 计算设备 |
| `--n_freqs` | `64` | S-Transform频率点数 |
| `--test_mode` | `False` | 测试模式 |

### Python API使用

```python
from data_loader import EEGDataLoader
from feature_extraction import ComprehensiveFeatureExtractor

# 加载数据
loader = EEGDataLoader(
    dataset_path='../dataset/ds004504',
    window_size=4.0,
    overlap=0.5
)

# 加载单个受试者
data, channel_names, label = loader.load_single_subject('sub-001')
segments = loader.segment_signal(data)

# 创建特征提取器
extractor = ComprehensiveFeatureExtractor(
    sample_rate=500.0,
    use_cnn=True
)

# 提取特征
features, feature_names = extractor.extract_all_features(
    segments[0], channel_names
)

print(f"特征数量: {len(feature_names)}")
```

## 输出文件

运行后在 `cache/` 目录下生成：

- `features_YYYYMMDD_HHMMSS.pkl` - 特征矩阵和标签
- `config_YYYYMMDD_HHMMSS.json` - 配置信息
- `splits_YYYYMMDD_HHMMSS.pkl` - 数据集划分信息

### 加载保存的特征

```python
import pickle

with open('cache/features_YYYYMMDD_HHMMSS.pkl', 'rb') as f:
    data = pickle.load(f)

features = data['features']       # 特征矩阵 (n_samples, n_features)
labels = data['labels']           # 标签 (n_samples,)
feature_names = data['feature_names']  # 特征名称列表
subject_ids = data['subject_ids']      # 受试者ID
```

## 项目结构

```
experiment_v1.0.0/
├── main.py                 # 主程序
├── data_loader.py          # 数据加载模块
├── s_transform.py          # S-Transform实现
├── traditional_features.py # 传统特征提取
├── cnn_feature_extractor.py# CNN特征提取
├── feature_extraction.py   # 综合特征提取
├── requirements.txt        # 依赖包
├── README.md              # 说明文档
└── cache/                 # 输出目录（自动创建）
```

## CNN架构

基于论文的 Stockwell-CNN 架构：

```
输入: S-Transform时频图 (channels × freqs × times)
    ↓
CNN Block 1: Conv2D(32) → BatchNorm → ReLU → MaxPool
    ↓
CNN Block 2: Conv2D(64) → BatchNorm → ReLU → MaxPool
    ↓
CNN Block 3: Conv2D(128) → BatchNorm → ReLU → MaxPool
    ↓
Attention Block: 注意力权重加权
    ↓
Flatten → FC(256) → Dropout → FC(128)
    ↓
输出: 特征向量 (128维)
```

## 注意事项

1. **内存需求**: 完整数据集处理需要较大内存，建议 ≥16GB RAM
2. **GPU加速**: CNN特征提取可使用GPU加速，设置 `--device cuda`
3. **数据预处理**: 默认使用预处理后的数据（`preprocess/`目录）
4. **数据划分**: 按受试者划分，避免数据泄露

## 标签说明

| 标签值 | 组别 | 描述 |
|--------|------|------|
| 0 | AD | 阿尔茨海默病 |
| 1 | FTD | 额颞叶痴呆 |
| 2 | CN | 认知正常（健康对照） |

## License

MIT License

