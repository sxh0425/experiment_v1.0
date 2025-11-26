# -*- coding: utf-8 -*-
"""
工具函数模块 - 评估指标、日志、可视化等
"""

import numpy as np
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

from config import LOGS_DIR, RESULTS_DIR, LOG_FORMAT, LOG_DATE_FORMAT


def compute_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    计算分类评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        
    Returns:
        Dict[str, float]: 包含ACC, SENS, SPEC, F1的字典
    """
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # 准确率 (Accuracy)
    acc = (tp + tn) / (tp + tn + fp + fn)
    
    # 敏感度 (Sensitivity / Recall / True Positive Rate)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # 特异度 (Specificity / True Negative Rate)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # F1分数
    f1 = f1_score(y_true, y_pred)
    
    return {
        'accuracy': acc,
        'sensitivity': sens,
        'specificity': spec,
        'f1_score': f1,
        'true_positive': int(tp),
        'true_negative': int(tn),
        'false_positive': int(fp),
        'false_negative': int(fn)
    }


def compute_metrics_from_cm(
    confusion_mat: np.ndarray
) -> Dict[str, float]:
    """
    从累积的混淆矩阵计算指标
    
    Args:
        confusion_mat: 2x2混淆矩阵
        
    Returns:
        Dict[str, float]: 评估指标
    """
    tn, fp, fn, tp = confusion_mat.ravel()
    
    acc = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * sens / (precision + sens) if (precision + sens) > 0 else 0.0
    
    return {
        'accuracy': acc,
        'sensitivity': sens,
        'specificity': spec,
        'f1_score': f1,
        'precision': precision,
        'true_positive': int(tp),
        'true_negative': int(tn),
        'false_positive': int(fp),
        'false_negative': int(fn)
    }


def setup_logger(
    name: str, 
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志器名称
        log_file: 日志文件名（可选）
        level: 日志级别
        
    Returns:
        logging.Logger: 配置好的日志器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除现有handler
    logger.handlers = []
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 文件handler
    if log_file:
        log_path = LOGS_DIR / log_file
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
    
    return logger


def save_results(
    results: Dict,
    filename: str,
    subdir: Optional[str] = None
):
    """
    保存结果到JSON文件
    
    Args:
        results: 结果字典
        filename: 文件名
        subdir: 子目录（可选）
    """
    if subdir:
        save_dir = RESULTS_DIR / subdir
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = RESULTS_DIR
    
    filepath = save_dir / filename
    
    # 转换numpy类型为Python类型
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj
    
    results = convert(results)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logging.info(f"结果已保存至: {filepath}")


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    title: str,
    save_path: Optional[Path] = None,
    normalize: bool = False
):
    """
    绘制混淆矩阵
    
    Args:
        cm: 混淆矩阵
        labels: 类别标签
        title: 图表标题
        save_path: 保存路径
        normalize: 是否归一化
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt=fmt, 
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        logging.info(f"混淆矩阵已保存至: {save_path}")
    
    plt.close()


def format_metrics_table(
    results: Dict[str, Dict[str, float]],
    task_name: str
) -> str:
    """
    格式化指标表格
    
    Args:
        results: {model_name: metrics_dict}
        task_name: 任务名称
        
    Returns:
        str: 格式化的表格字符串
    """
    header = f"\n{'='*70}\n{task_name} 分类结果\n{'='*70}"
    
    col_headers = f"{'模型':<15} {'ACC':>10} {'SENS':>10} {'SPEC':>10} {'F1':>10}"
    separator = "-" * 70
    
    rows = []
    for model_name, metrics in results.items():
        row = (f"{model_name:<15} "
               f"{metrics['accuracy']*100:>9.2f}% "
               f"{metrics['sensitivity']*100:>9.2f}% "
               f"{metrics['specificity']*100:>9.2f}% "
               f"{metrics['f1_score']*100:>9.2f}%")
        rows.append(row)
    
    table = "\n".join([header, col_headers, separator] + rows + [separator])
    
    return table


def get_timestamp() -> str:
    """获取当前时间戳字符串"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class ResultsLogger:
    """结果记录器类"""
    
    def __init__(self, experiment_name: str):
        """
        初始化结果记录器
        
        Args:
            experiment_name: 实验名称
        """
        self.experiment_name = experiment_name
        self.timestamp = get_timestamp()
        self.results = {
            'experiment_name': experiment_name,
            'timestamp': self.timestamp,
            'tasks': {}
        }
        
        # 创建日志器
        log_file = f"{experiment_name}_{self.timestamp}.log"
        self.logger = setup_logger(experiment_name, log_file)
        
    def add_task_result(
        self, 
        task_name: str, 
        model_name: str, 
        metrics: Dict[str, float],
        confusion_mat: Optional[np.ndarray] = None
    ):
        """
        添加任务结果
        
        Args:
            task_name: 任务名称
            model_name: 模型名称
            metrics: 评估指标
            confusion_mat: 混淆矩阵
        """
        if task_name not in self.results['tasks']:
            self.results['tasks'][task_name] = {}
        
        self.results['tasks'][task_name][model_name] = {
            'metrics': metrics,
            'confusion_matrix': confusion_mat.tolist() if confusion_mat is not None else None
        }
        
        self.logger.info(
            f"{task_name} - {model_name}: "
            f"ACC={metrics['accuracy']*100:.2f}%, "
            f"F1={metrics['f1_score']*100:.2f}%"
        )
    
    def save(self, filename: Optional[str] = None):
        """保存所有结果"""
        if filename is None:
            filename = f"results_{self.experiment_name}_{self.timestamp}.json"
        
        save_results(self.results, filename)
    
    def print_summary(self):
        """打印结果摘要"""
        for task_name, models in self.results['tasks'].items():
            model_metrics = {
                model: data['metrics'] 
                for model, data in models.items()
            }
            print(format_metrics_table(model_metrics, task_name))


if __name__ == "__main__":
    # 测试工具函数
    print("测试工具函数模块...")
    
    # 测试指标计算
    y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 1, 1, 0])
    
    metrics = compute_metrics(y_true, y_pred)
    print("\n评估指标测试:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # 测试结果记录器
    print("\n结果记录器测试:")
    logger = ResultsLogger("test_experiment")
    logger.add_task_result("AD_CN", "test_model", metrics)
    logger.print_summary()

