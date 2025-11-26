# -*- coding: utf-8 -*-
"""
主程序 - EEG分类论文复现
论文: A Dataset of Scalp EEG Recordings of Alzheimer's Disease, 
      Frontotemporal Dementia and Healthy Subjects from Routine EEG

用法:
    # 运行所有模型和所有任务
    python main.py
    
    # 选择特定模型
    python main.py --models lightgbm svm
    
    # 选择特定任务
    python main.py --tasks AD_CN
    
    # 单独运行一个模型
    python main.py --models knn --tasks FTD_CN
    
    # 跳过特征提取（使用缓存）
    python main.py --use-cache
    
    # 禁用Hyperopt优化（加快LightGBM）
    python main.py --no-hyperopt
"""

import argparse
import sys
import logging
import numpy as np
from datetime import datetime
from pathlib import Path

from config import (
    AVAILABLE_MODELS, 
    CLASSIFICATION_TASKS,
    RESULTS_DIR,
    RANDOM_SEED
)
from data_loader import load_all_subjects_data, describe_dataset
from feature_extraction import extract_features_all_subjects
from loso_validation import run_classification_task
from utils import (
    ResultsLogger,
    save_results,
    plot_confusion_matrix,
    format_metrics_table,
    get_timestamp
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='EEG分类论文复现 - AD/FTD vs CN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                           # 运行全部
  python main.py --models lightgbm svm     # 只运行LightGBM和SVM
  python main.py --tasks AD_CN             # 只运行AD vs CN任务
  python main.py --models knn --tasks FTD_CN  # 指定模型和任务
  python main.py --use-cache               # 使用缓存的特征
  python main.py --no-hyperopt             # 禁用LightGBM的Hyperopt优化
        """
    )
    
    parser.add_argument(
        '--models', 
        nargs='+', 
        choices=AVAILABLE_MODELS,
        default=None,
        help=f'要运行的模型，可选: {AVAILABLE_MODELS}。默认运行全部。'
    )
    
    parser.add_argument(
        '--tasks',
        nargs='+',
        choices=list(CLASSIFICATION_TASKS.keys()),
        default=None,
        help=f'要运行的任务，可选: {list(CLASSIFICATION_TASKS.keys())}。默认运行全部。'
    )
    
    parser.add_argument(
        '--use-cache',
        action='store_true',
        help='使用缓存的特征数据（如果存在）'
    )
    
    parser.add_argument(
        '--no-hyperopt',
        action='store_true',
        help='禁用LightGBM的Hyperopt超参数优化（加快速度）'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细输出'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='结果输出目录'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    np.random.seed(RANDOM_SEED)
    
    # 确定要运行的模型和任务
    models_to_run = args.models if args.models else AVAILABLE_MODELS
    tasks_to_run = args.tasks if args.tasks else list(CLASSIFICATION_TASKS.keys())
    
    # 设置输出目录
    timestamp = get_timestamp()
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = RESULTS_DIR / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    log_file = output_dir / f"experiment_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    logger = logging.getLogger(__name__)
    
    # 打印运行配置
    print("\n" + "="*70)
    print("EEG分类论文复现")
    print("="*70)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模型: {models_to_run}")
    print(f"任务: {tasks_to_run}")
    print(f"输出目录: {output_dir}")
    print("="*70 + "\n")
    
    # 描述数据集
    describe_dataset()
    
    # 加载数据
    print("\n[1/4] 加载EEG数据...")
    # 只加载需要的组别
    groups_needed = set()
    for task in tasks_to_run:
        task_config = CLASSIFICATION_TASKS[task]
        groups_needed.add(task_config['positive'])
        groups_needed.add(task_config['negative'])
    
    subjects_data = load_all_subjects_data(groups=list(groups_needed))
    print(f"      已加载 {len(subjects_data)} 个被试的数据")
    
    # 提取特征
    # 使用每个通道独立的特征：19通道 × 5频带 = 95个特征
    # 按通道排列: [ch1的5频带, ch2的5频带, ...]
    print("\n[2/4] 提取特征 (RBP - 95维: 按通道排列)...")
    cache_name = f"features_{'_'.join(sorted(groups_needed))}_95dim_bychannel"
    features_dict, labels_dict = extract_features_all_subjects(
        subjects_data,
        use_cache=args.use_cache,
        cache_name=cache_name,
        aggregate_channels=False,  # 不对通道平均
        feature_order='by_channel'  # 按通道排列特征
    )
    
    total_epochs = sum(len(f) for f in features_dict.values())
    n_features = list(features_dict.values())[0].shape[1] if features_dict else 0
    print(f"      共 {total_epochs} 个epochs，每个 {n_features} 个特征")
    
    # 存储所有结果
    all_results = {
        'timestamp': timestamp,
        'models': models_to_run,
        'tasks': tasks_to_run,
        'results': {}
    }
    
    # 运行分类任务
    print("\n[3/4] 运行LOSO交叉验证...")
    
    for task_name in tasks_to_run:
        task_config = CLASSIFICATION_TASKS[task_name]
        
        print(f"\n{'='*60}")
        print(f"任务: {task_config['name']}")
        print(f"{'='*60}")
        
        # 更新LightGBM的hyperopt设置
        if 'lightgbm' in models_to_run and args.no_hyperopt:
            # 临时修改LightGBM不使用hyperopt
            from models.lightgbm_model import LightGBMClassifier
            original_init = LightGBMClassifier.__init__
            def patched_init(self, *a, **kw):
                kw['use_hyperopt'] = False
                return original_init(self, *a, **kw)
            LightGBMClassifier.__init__ = patched_init
        
        task_results = run_classification_task(
            features_dict=features_dict,
            labels_dict=labels_dict,
            positive_label=task_config['positive'],
            negative_label=task_config['negative'],
            task_name=task_config['name'],
            models_to_run=models_to_run,
            verbose=True
        )
        
        all_results['results'][task_name] = task_results
        
        # 保存混淆矩阵图
        for model_name, model_results in task_results.items():
            cm = model_results['confusion_matrix']
            labels = [task_config['negative'], task_config['positive']]
            
            # 原始混淆矩阵
            plot_confusion_matrix(
                cm, 
                labels,
                f"{task_config['name']} - {model_name}",
                output_dir / f"cm_{task_name}_{model_name}.png"
            )
            
            # 归一化混淆矩阵
            plot_confusion_matrix(
                cm,
                labels,
                f"{task_config['name']} - {model_name} (Normalized)",
                output_dir / f"cm_{task_name}_{model_name}_normalized.png",
                normalize=True
            )
    
    # 打印汇总结果
    print("\n[4/4] 结果汇总")
    print("\n" + "="*70)
    print("实验结果汇总")
    print("="*70)
    
    for task_name in tasks_to_run:
        task_config = CLASSIFICATION_TASKS[task_name]
        task_results = all_results['results'][task_name]
        
        # 格式化表格
        model_metrics = {
            model: results['metrics'] 
            for model, results in task_results.items()
        }
        print(format_metrics_table(model_metrics, task_config['name']))
    
    # 与论文结果对比
    print("\n" + "="*70)
    print("论文报告结果 (参考)")
    print("="*70)
    
    paper_results = {
        'AD_CN': {
            'LightGBM': {'ACC': 76.43, 'SENS': 76.01, 'SPEC': 76.16, 'F1': 76.12},
            'SVM': {'ACC': 73.14, 'SENS': 71.89, 'SPEC': 75.98, 'F1': 73.74},
            'kNN': {'ACC': 71.23, 'SENS': 69.67, 'SPEC': 74.19, 'F1': 72.81},
            'MLP': {'ACC': 73.12, 'SENS': 73.00, 'SPEC': 74.63, 'F1': 74.82},
            'Random Forest': {'ACC': 77.01, 'SENS': 78.32, 'SPEC': 80.94, 'F1': 75.31}
        },
        'FTD_CN': {
            'LightGBM': {'ACC': 72.43, 'SENS': 61.13, 'SPEC': 80.74, 'F1': 67.32},
            'SVM': {'ACC': 70.14, 'SENS': 62.41, 'SPEC': 75.98, 'F1': 68.32},
            'kNN': {'ACC': 67.34, 'SENS': 59.67, 'SPEC': 76.13, 'F1': 70.81},
            'MLP': {'ACC': 73.12, 'SENS': 63.00, 'SPEC': 78.63, 'F1': 72.82},
            'Random Forest': {'ACC': 72.01, 'SENS': 72.32, 'SPEC': 80.94, 'F1': 66.31}
        }
    }
    
    for task_name in tasks_to_run:
        if task_name in paper_results:
            print(f"\n{CLASSIFICATION_TASKS[task_name]['name']}:")
            print(f"{'模型':<15} {'ACC':>10} {'SENS':>10} {'SPEC':>10} {'F1':>10}")
            print("-" * 60)
            for model, metrics in paper_results[task_name].items():
                print(f"{model:<15} {metrics['ACC']:>9.2f}% {metrics['SENS']:>9.2f}% "
                      f"{metrics['SPEC']:>9.2f}% {metrics['F1']:>9.2f}%")
    
    # 保存结果
    save_results(all_results, f"results_{timestamp}.json", subdir=None)
    
    # 保存到指定输出目录
    results_file = output_dir / "results.json"
    save_results(all_results, results_file.name, subdir=output_dir.name)
    
    print(f"\n结果已保存至: {output_dir}")
    print("\n完成！")
    
    return all_results


if __name__ == "__main__":
    main()
