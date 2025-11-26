# -*- coding: utf-8 -*-
"""
MLP分类器 - 多层感知机
论文设置: MLP (1 hidden layer of 3 neurons)
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .base_model import BaseClassifier


class MLPClassifierWrapper(BaseClassifier):
    """多层感知机分类器"""
    
    def __init__(
        self,
        hidden_layer_sizes: tuple = (3,),
        activation: str = 'relu',
        solver: str = 'adam',
        max_iter: int = 1000,
        random_state: int = 42,
        early_stopping: bool = True,
        validation_fraction: float = 0.1
    ):
        """
        初始化
        
        Args:
            hidden_layer_sizes: 隐藏层结构，论文使用(3,)即1层3个神经元
            activation: 激活函数
            solver: 优化器
            max_iter: 最大迭代次数
            random_state: 随机种子
            early_stopping: 是否使用早停
            validation_fraction: 验证集比例
        """
        super().__init__("MLP")
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.max_iter = max_iter
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MLPClassifierWrapper':
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
            ('mlp', MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                solver=self.solver,
                max_iter=self.max_iter,
                random_state=self.random_state,
                early_stopping=self.early_stopping,
                validation_fraction=self.validation_fraction
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

