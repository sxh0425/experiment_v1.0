# EEG分类论文复现

## 论文信息

**标题**: A Dataset of Scalp EEG Recordings of Alzheimer's Disease, Frontotemporal Dementia and Healthy Subjects from Routine EEG

**DOI**: 10.18112/openneuro.ds004504.v1.0.2

## 数据集描述

| 项目 | 说明 |
|------|------|
| 被试数 | 88人 (AD: 36, FTD: 23, CN: 29) |
| 采样率 | 500 Hz |
| 电极数 | 19个头皮电极 (10-20国际系统) |
| 记录状态 | 静息态闭眼 |

## 方法复现

### 特征提取
- **时间窗口**: 4秒，50%重叠
- **特征**: 5频带相对功率 (RBP)
  - Delta: 0.5-4 Hz
  - Theta: 4-8 Hz
  - Alpha: 8-13 Hz
  - Beta: 13-25 Hz
  - Gamma: 25-45 Hz

### 分类模型
1. **LightGBM** - Hyperopt超参数优化
2. **SVM** - 多项式核
3. **kNN** - k=3
4. **MLP** - 1隐藏层，3个神经元
5. **Random Forest**

### 验证方法
- **Leave-One-Subject-Out (LOSO)** 交叉验证

### 评估指标
- 准确率 (Accuracy, ACC)
- 敏感度 (Sensitivity, SENS)
- 特异度 (Specificity, SPEC)
- F1分数

## 项目结构

```
dataset_paper/
├── config.py               # 配置文件
├── data_loader.py          # 数据加载模块
├── feature_extraction.py   # 特征提取模块
├── loso_validation.py      # LOSO验证框架
├── utils.py                # 工具函数
├── main.py                 # 主运行程序
├── requirements.txt        # 依赖包
├── models/                 # 模型模块
│   ├── __init__.py
│   ├── base_model.py
│   ├── lightgbm_model.py
│   ├── svm_model.py
│   ├── knn_model.py
│   ├── mlp_model.py
│   └── random_forest_model.py
├── results/                # 结果输出
│   └── logs/               # 日志文件
└── cache/                  # 特征缓存
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 运行所有模型和任务

```bash
python main.py
```

### 选择特定模型

```bash
# 只运行LightGBM和SVM
python main.py --models lightgbm svm

# 只运行kNN
python main.py --models knn
```

### 选择特定任务

```bash
# 只运行AD vs CN分类
python main.py --tasks AD_CN

# 只运行FTD vs CN分类
python main.py --tasks FTD_CN
```

### 组合使用

```bash
# 使用kNN模型运行FTD vs CN任务
python main.py --models knn --tasks FTD_CN

# 运行多个模型和任务
python main.py --models lightgbm random_forest --tasks AD_CN FTD_CN
```

### 其他选项

```bash
# 使用缓存的特征（加快速度）
python main.py --use-cache

# 禁用LightGBM的Hyperopt优化（加快速度）
python main.py --no-hyperopt

# 显示详细输出
python main.py --verbose

# 指定输出目录
python main.py --output-dir ./my_results
```

## 可用模型名称

| 模型名称 | 说明 |
|----------|------|
| `lightgbm` | LightGBM (Hyperopt优化) |
| `svm` | SVM (多项式核) |
| `knn` | k近邻 (k=3) |
| `mlp` | 多层感知机 (1层3神经元) |
| `random_forest` | 随机森林 |

## 可用任务名称

| 任务名称 | 说明 |
|----------|------|
| `AD_CN` | 阿尔茨海默病 vs 健康对照 |
| `FTD_CN` | 额颞叶痴呆 vs 健康对照 |

## 论文报告结果

### AD vs CN

| 模型 | ACC | SENS | SPEC | F1 |
|------|-----|------|------|-----|
| LightGBM | 76.43% | 76.01% | 76.16% | 76.12% |
| SVM | 73.14% | 71.89% | 75.98% | 73.74% |
| kNN | 71.23% | 69.67% | 74.19% | 72.81% |
| MLP | 73.12% | 73.00% | 74.63% | 74.82% |
| Random Forests | 77.01% | 78.32% | 80.94% | 75.31% |

### FTD vs CN

| 模型 | ACC | SENS | SPEC | F1 |
|------|-----|------|------|-----|
| LightGBM | 72.43% | 61.13% | 80.74% | 67.32% |
| SVM | 70.14% | 62.41% | 75.98% | 68.32% |
| kNN | 67.34% | 59.67% | 76.13% | 70.81% |
| MLP | 73.12% | 63.00% | 78.63% | 72.82% |
| Random Forests | 72.01% | 72.32% | 80.94% | 66.31% |

## 输出文件

运行后会在 `results/` 目录下生成：

- `results_YYYYMMDD_HHMMSS.json` - JSON格式的详细结果
- `experiment_YYYYMMDD_HHMMSS.log` - 实验日志
- `cm_TASK_MODEL.png` - 混淆矩阵图
- `cm_TASK_MODEL_normalized.png` - 归一化混淆矩阵图

## 注意事项

1. 首次运行需要加载所有EEG数据，可能需要几分钟
2. 特征提取后会自动缓存，后续运行可使用 `--use-cache` 加速
3. LightGBM的Hyperopt优化可能较慢，可用 `--no-hyperopt` 跳过
4. 确保预处理后的数据位于 `../dataset/ds004504/preprocess/` 目录

## 参考文献

Miltiadous, A. et al. (2023). A Dataset of Scalp EEG Recordings of Alzheimer's Disease, Frontotemporal Dementia and Healthy Subjects from Routine EEG. *Data*, 8, 95.

