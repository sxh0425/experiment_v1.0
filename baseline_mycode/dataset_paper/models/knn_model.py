# -*- coding: utf-8 -*-
"""
kNN分类器
论文设置: kNN (k=3)
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

from .base_model import BaseClassifier


class KNNClassifier(BaseClassifier):
    """k近邻分类器"""
    
    def __init__(
        self,
        n_neighbors: int = 3,
        weights: str = 'uniform',
        metric: str = 'minkowski',
        scaler_type: str = 'none'  # 'standard', 'minmax', 'none'
    ):
        """
        初始化
        
        Args:
            n_neighbors: 近邻数量
            weights: 权重类型 ('uniform', 'distance')
            metric: 距离度量
            scaler_type: 标准化类型
        """
        super().__init__("kNN")
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.scaler_type = scaler_type
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNNClassifier':
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 标签数组
            
        Returns:
            self
        """
        # 根据scaler_type选择标准化方式
        if self.scaler_type == 'standard':
            steps = [
                ('scaler', StandardScaler()),
                ('knn', KNeighborsClassifier(
                    n_neighbors=self.n_neighbors,
                    weights=self.weights,
                    metric=self.metric
                ))
            ]
        elif self.scaler_type == 'minmax':
            steps = [
                ('scaler', MinMaxScaler()),
                ('knn', KNeighborsClassifier(
                    n_neighbors=self.n_neighbors,
                    weights=self.weights,
                    metric=self.metric
                ))
            ]
        else:  # 'none' - RBP特征本身已归一化
            steps = [
                ('knn', KNeighborsClassifier(
                    n_neighbors=self.n_neighbors,
                    weights=self.weights,
                    metric=self.metric
                ))
            ]
        
        self.model = Pipeline(steps)
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

