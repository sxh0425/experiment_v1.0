# -*- coding: utf-8 -*-
"""
配置文件 - EEG分类论文复现
论文: A Dataset of Scalp EEG Recordings of Alzheimer's Disease, 
      Frontotemporal Dementia and Healthy Subjects from Routine EEG
"""

import os
from pathlib import Path

# ============== 路径配置 ==============
# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 数据集路径
DATASET_ROOT = PROJECT_ROOT.parent / "dataset" / "ds004504"
PREPROCESS_DIR = DATASET_ROOT / "preprocess"
PARTICIPANTS_FILE = DATASET_ROOT / "participants.tsv"

# 输出路径
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = RESULTS_DIR / "logs"
CACHE_DIR = PROJECT_ROOT / "cache"

# 创建必要的目录
for dir_path in [RESULTS_DIR, LOGS_DIR, CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============== EEG参数 ==============
# 采样率 (Hz)
SAMPLING_RATE = 500

# 频带定义 (Hz)
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 25),
    'gamma': (25, 45)
}

# 滤波范围
FILTER_LOW = 0.5
FILTER_HIGH = 45

# ============== 特征提取参数 ==============
# 时间窗口 (秒)
WINDOW_SIZE = 4.0

# 重叠比例
OVERLAP_RATIO = 0.5

# Welch方法参数
WELCH_NPERSEG = 256  # 每段长度

# ============== 被试组别 ==============
GROUP_LABELS = {
    'A': 'AD',    # Alzheimer's Disease
    'F': 'FTD',   # Frontotemporal Dementia  
    'C': 'CN'     # Healthy Control
}

# 分类任务
CLASSIFICATION_TASKS = {
    'AD_CN': {'positive': 'A', 'negative': 'C', 'name': 'AD vs CN'},
    'FTD_CN': {'positive': 'F', 'negative': 'C', 'name': 'FTD vs CN'}
}

# ============== 模型参数 ==============
# 可用的模型列表
AVAILABLE_MODELS = ['lightgbm', 'svm', 'knn', 'mlp', 'random_forest']

# kNN参数
KNN_K = 3

# MLP参数 (1个隐藏层, 3个神经元)
MLP_HIDDEN_LAYERS = (3,)
MLP_MAX_ITER = 1000

# SVM参数 (多项式核)
SVM_KERNEL = 'poly'

# Random Forest参数
RF_N_ESTIMATORS = 100
RF_RANDOM_STATE = 42

# LightGBM 基础参数 (会被Hyperopt优化)
LGBM_BASE_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'verbosity': -1,
    'random_state': 42
}

# Hyperopt优化迭代次数
HYPEROPT_MAX_EVALS = 50

# ============== 随机种子 ==============
RANDOM_SEED = 42

# ============== 日志配置 ==============
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

