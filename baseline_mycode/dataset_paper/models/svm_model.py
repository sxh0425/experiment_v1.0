# -*- coding: utf-8 -*-
"""
SVM分类器 - 多项式核
论文设置: SVM (polynomial kernel)
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .base_model import BaseClassifier


class SVMClassifier(BaseClassifier):
    """SVM分类器，使用多项式核"""
    
    def __init__(
        self,
        kernel: str = 'poly',
        degree: int = 3,
        C: float = 1.0,
        gamma: str = 'scale',
        random_state: int = 42
    ):
        """
        初始化
        
        Args:
            kernel: 核函数类型 ('poly', 'rbf', 'linear', etc.)
            degree: 多项式核的度数
            C: 正则化参数
            gamma: 核系数
            random_state: 随机种子
        """
        super().__init__("SVM")
        self.kernel = kernel
        self.degree = degree
        self.C = C
        self.gamma = gamma
        self.random_state = random_state
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVMClassifier':
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 标签数组
            
        Returns:
            self
        """
        # 使用Pipeline包含标准化
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(
                kernel=self.kernel,
                degree=self.degree,
                C=self.C,
                gamma=self.gamma,
                random_state=self.random_state,
                probability=True  # 启用概率估计
            ))
        ])
        
        self.model.fit(X, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise RuntimeError("模型尚未训练")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        if not self.is_fitted:
            raise RuntimeError("模型尚未训练")
        return self.model.predict_proba(X)

