# -*- coding: utf-8 -*-
"""
基础模型类 - 定义统一的接口
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional


class BaseClassifier(ABC):
    """基础分类器抽象类"""
    
    def __init__(self, name: str):
        """
        初始化
        
        Args:
            name: 模型名称
        """
        self.name = name
        self.model = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseClassifier':
        """
        训练模型
        
        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签数组 (n_samples,)
            
        Returns:
            self
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X: 特征矩阵 (n_samples, n_features)
            
        Returns:
            np.ndarray: 预测标签
        """
        pass
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        预测概率（如果支持）
        
        Args:
            X: 特征矩阵
            
        Returns:
            np.ndarray: 预测概率，形状为 (n_samples, n_classes)
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"

