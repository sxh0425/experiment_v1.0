# -*- coding: utf-8 -*-
"""
LOSO验证模块 - Leave-One-Subject-Out交叉验证
论文方法: 每次留出一个被试的所有epochs作为测试集
"""

import numpy as np
from typing import Dict, List, Tuple, Callable
from sklearn.metrics import confusion_matrix
import logging
from tqdm import tqdm

from utils import compute_metrics_from_cm

logger = logging.getLogger(__name__)


def leave_one_subject_out_cv(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    model_factory: Callable,
    verbose: bool = True
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    执行Leave-One-Subject-Out交叉验证
    
    Args:
        X: 特征矩阵 (n_samples, n_features)
        y: 标签数组 (n_samples,)
        subject_ids: 每个样本对应的被试ID (n_samples,)
        model_factory: 创建模型的函数，调用时返回新的模型实例
        verbose: 是否显示进度
        
    Returns:
        Tuple[Dict[str, float], np.ndarray]:
            - 评估指标字典 {'accuracy', 'sensitivity', 'specificity', 'f1_score'}
            - 累积的混淆矩阵
    """
    unique_subjects = np.unique(subject_ids)
    n_subjects = len(unique_subjects)
    
    logger.info(f"开始LOSO验证: {n_subjects} 个被试")
    
    # 累积混淆矩阵
    cumulative_cm = np.zeros((2, 2), dtype=int)
    
    # 记录每个fold的结果
    fold_results = []
    
    iterator = tqdm(unique_subjects, desc="LOSO") if verbose else unique_subjects
    
    for test_subject in iterator:
        # 分割训练集和测试集
        test_mask = subject_ids == test_subject
        train_mask = ~test_mask
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        # 检查测试集是否有样本
        if len(X_test) == 0:
            logger.warning(f"被试 {test_subject} 没有测试样本，跳过")
            continue
        
        # 创建新的模型实例
        model = model_factory()
        
        # 训练
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            logger.error(f"训练失败 (被试 {test_subject}): {e}")
            continue
        
        # 预测
        try:
            y_pred = model.predict(X_test)
        except Exception as e:
            logger.error(f"预测失败 (被试 {test_subject}): {e}")
            continue
        
        # 计算混淆矩阵并累加
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        cumulative_cm += cm
        
        # 记录此fold的准确率
        fold_acc = np.sum(y_pred == y_test) / len(y_test)
        fold_results.append({
            'subject': test_subject,
            'accuracy': fold_acc,
            'n_test': len(y_test)
        })
    
    # 从累积混淆矩阵计算最终指标
    metrics = compute_metrics_from_cm(cumulative_cm)
    
    # 添加额外统计信息
    metrics['n_subjects'] = len(fold_results)
    metrics['n_total_samples'] = int(np.sum([f['n_test'] for f in fold_results]))
    
    logger.info(f"LOSO完成: ACC={metrics['accuracy']*100:.2f}%, "
                f"F1={metrics['f1_score']*100:.2f}%")
    
    return metrics, cumulative_cm


def run_classification_task(
    features_dict: Dict[str, np.ndarray],
    labels_dict: Dict[str, str],
    positive_label: str,
    negative_label: str,
    task_name: str,
    models_to_run: List[str],
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    运行分类任务
    
    Args:
        features_dict: 特征字典 {subject_id: features_array}
        labels_dict: 标签字典 {subject_id: group_label}
        positive_label: 正类标签
        negative_label: 负类标签
        task_name: 任务名称
        models_to_run: 要运行的模型列表
        verbose: 是否显示详情
        
    Returns:
        Dict[str, Dict]: {model_name: {'metrics': ..., 'confusion_matrix': ...}}
    """
    from feature_extraction import prepare_data_for_task
    from models import get_model
    
    # 准备数据
    X, y, subject_ids = prepare_data_for_task(
        features_dict, labels_dict, positive_label, negative_label
    )
    
    logger.info(f"\n{'='*60}")
    logger.info(f"任务: {task_name}")
    logger.info(f"数据: {len(X)} 样本, {len(np.unique(subject_ids))} 被试")
    logger.info(f"类别分布: 正类={np.sum(y==1)}, 负类={np.sum(y==0)}")
    logger.info(f"{'='*60}")
    
    results = {}
    
    for model_name in models_to_run:
        logger.info(f"\n--- 模型: {model_name} ---")
        
        # 定义模型工厂函数
        def model_factory(name=model_name):
            return get_model(name)
        
        # 运行LOSO验证
        metrics, cm = leave_one_subject_out_cv(
            X, y, subject_ids, model_factory, verbose
        )
        
        results[model_name] = {
            'metrics': metrics,
            'confusion_matrix': cm
        }
        
        # 打印结果
        logger.info(f"  准确率 (ACC): {metrics['accuracy']*100:.2f}%")
        logger.info(f"  敏感度 (SENS): {metrics['sensitivity']*100:.2f}%")
        logger.info(f"  特异度 (SPEC): {metrics['specificity']*100:.2f}%")
        logger.info(f"  F1分数: {metrics['f1_score']*100:.2f}%")
    
    return results


if __name__ == "__main__":
    # 测试LOSO验证
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    print("测试LOSO验证模块...")
    
    # 创建测试数据
    np.random.seed(42)
    n_subjects = 10
    n_samples_per_subject = 50
    n_features = 5
    
    X = np.random.randn(n_subjects * n_samples_per_subject, n_features)
    y = np.array([i % 2 for i in range(n_subjects) for _ in range(n_samples_per_subject)])
    subject_ids = np.array([f"sub-{i:03d}" for i in range(n_subjects) 
                           for _ in range(n_samples_per_subject)])
    
    # 测试验证
    from models import get_model
    
    def model_factory():
        return get_model('knn')
    
    metrics, cm = leave_one_subject_out_cv(X, y, subject_ids, model_factory)
    
    print(f"\n测试结果:")
    print(f"  ACC: {metrics['accuracy']*100:.2f}%")
    print(f"  混淆矩阵:\n{cm}")

