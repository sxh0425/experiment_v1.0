# -*- coding: utf-8 -*-
"""
Random Forest分类器
论文设置: Random Forests
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .base_model import BaseClassifier


class RandomForestClassifierWrapper(BaseClassifier):
    """随机森林分类器"""
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        初始化
        
        Args:
            n_estimators: 树的数量
            max_depth: 最大深度
            min_samples_split: 分裂所需最小样本数
            min_samples_leaf: 叶节点最小样本数
            random_state: 随机种子
            n_jobs: 并行作业数
        """
        super().__init__("Random Forest")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestClassifierWrapper':
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 标签数组
            
        Returns:
            self
        """
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
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
    
    def get_feature_importance(self) -> np.ndarray:
        """获取特征重要性"""
        if not self.is_fitted:
            raise RuntimeError("模型尚未训练")
        return self.model.feature_importances_

