# -*- coding: utf-8 -*-
"""
LightGBM分类器 - 使用Hyperopt进行超参数优化
论文设置: LightGBM (hyperparameter optimized by Hyperopt)
"""

import numpy as np
import lightgbm as lgb
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.model_selection import cross_val_score
import logging
import warnings

from .base_model import BaseClassifier

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class LightGBMClassifier(BaseClassifier):
    """LightGBM分类器，带Hyperopt超参数优化"""
    
    def __init__(
        self, 
        use_hyperopt: bool = True,
        max_evals: int = 50,
        random_state: int = 42,
        verbose: bool = False
    ):
        """
        初始化
        
        Args:
            use_hyperopt: 是否使用Hyperopt优化
            max_evals: Hyperopt最大评估次数
            random_state: 随机种子
            verbose: 是否显示详细信息
        """
        super().__init__("LightGBM")
        self.use_hyperopt = use_hyperopt
        self.max_evals = max_evals
        self.random_state = random_state
        self.verbose = verbose
        self.best_params = None
        
    def _get_search_space(self):
        """定义超参数搜索空间"""
        return {
            'num_leaves': hp.choice('num_leaves', range(20, 150)),
            'max_depth': hp.choice('max_depth', range(3, 12)),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
            'n_estimators': hp.choice('n_estimators', range(50, 300)),
            'min_child_samples': hp.choice('min_child_samples', range(10, 50)),
            'subsample': hp.uniform('subsample', 0.6, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
            'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-8), np.log(10.0)),
            'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-8), np.log(10.0)),
        }
    
    def _objective(self, params, X, y):
        """Hyperopt目标函数"""
        model = lgb.LGBMClassifier(
            **params,
            objective='binary',
            random_state=self.random_state,
            verbosity=-1,
            force_col_wise=True
        )
        
        # 使用3折交叉验证评估
        try:
            scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
            accuracy = scores.mean()
        except Exception as e:
            logger.debug(f"评估失败: {e}")
            accuracy = 0.0
        
        return {'loss': -accuracy, 'status': STATUS_OK}
    
    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray):
        """使用Hyperopt优化超参数"""
        if self.verbose:
            logger.info("开始Hyperopt超参数优化...")
        
        trials = Trials()
        search_space = self._get_search_space()
        
        best = fmin(
            fn=lambda params: self._objective(params, X, y),
            space=search_space,
            algo=tpe.suggest,
            max_evals=self.max_evals,
            trials=trials,
            verbose=False,
            rstate=np.random.default_rng(self.random_state)
        )
        
        # 转换choice参数
        self.best_params = {
            'num_leaves': list(range(20, 150))[best['num_leaves']],
            'max_depth': list(range(3, 12))[best['max_depth']],
            'learning_rate': best['learning_rate'],
            'n_estimators': list(range(50, 300))[best['n_estimators']],
            'min_child_samples': list(range(10, 50))[best['min_child_samples']],
            'subsample': best['subsample'],
            'colsample_bytree': best['colsample_bytree'],
            'reg_alpha': best['reg_alpha'],
            'reg_lambda': best['reg_lambda'],
        }
        
        if self.verbose:
            logger.info(f"最佳参数: {self.best_params}")
        
        return self.best_params
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LightGBMClassifier':
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 标签数组
            
        Returns:
            self
        """
        if self.use_hyperopt:
            params = self._optimize_hyperparameters(X, y)
        else:
            params = {
                'num_leaves': 31,
                'max_depth': -1,
                'learning_rate': 0.1,
                'n_estimators': 100
            }
        
        self.model = lgb.LGBMClassifier(
            **params,
            objective='binary',
            random_state=self.random_state,
            verbosity=-1,
            force_col_wise=True
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

