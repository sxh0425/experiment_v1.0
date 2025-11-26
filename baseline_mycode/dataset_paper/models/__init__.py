# -*- coding: utf-8 -*-
"""
模型模块 - 包含5种分类器
- LightGBM (Hyperopt优化)
- SVM (多项式核)
- kNN (k=3)
- MLP (1隐藏层, 3神经元)
- Random Forest
"""

from .lightgbm_model import LightGBMClassifier
from .svm_model import SVMClassifier
from .knn_model import KNNClassifier
from .mlp_model import MLPClassifierWrapper
from .random_forest_model import RandomForestClassifierWrapper

__all__ = [
    'LightGBMClassifier',
    'SVMClassifier', 
    'KNNClassifier',
    'MLPClassifierWrapper',
    'RandomForestClassifierWrapper'
]

# 模型名称映射
MODEL_CLASSES = {
    'lightgbm': LightGBMClassifier,
    'svm': SVMClassifier,
    'knn': KNNClassifier,
    'mlp': MLPClassifierWrapper,
    'random_forest': RandomForestClassifierWrapper
}


def get_model(model_name: str, **kwargs):
    """
    获取指定名称的模型实例
    
    Args:
        model_name: 模型名称
        **kwargs: 模型参数
        
    Returns:
        模型实例
    """
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"未知模型: {model_name}. 可选: {list(MODEL_CLASSES.keys())}")
    
    return MODEL_CLASSES[model_name](**kwargs)
