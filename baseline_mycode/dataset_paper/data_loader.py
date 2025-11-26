# -*- coding: utf-8 -*-
"""
数据加载模块 - 加载EEG数据和被试信息
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import mne
import logging

from config import (
    PREPROCESS_DIR, 
    PARTICIPANTS_FILE, 
    GROUP_LABELS,
    SAMPLING_RATE
)

# 设置日志
logger = logging.getLogger(__name__)


def load_participants_info() -> pd.DataFrame:
    """
    加载被试信息表
    
    Returns:
        pd.DataFrame: 包含participant_id, Gender, Age, Group, MMSE的DataFrame
    """
    df = pd.read_csv(PARTICIPANTS_FILE, sep='\t')
    logger.info(f"加载被试信息: {len(df)} 个被试")
    
    # 统计各组人数
    group_counts = df['Group'].value_counts()
    for group, count in group_counts.items():
        group_name = GROUP_LABELS.get(group, group)
        logger.info(f"  {group_name} ({group}): {count} 人")
    
    return df


def load_eeg_data(subject_id: str) -> Tuple[np.ndarray, float, List[str]]:
    """
    加载单个被试的EEG数据
    
    Args:
        subject_id: 被试ID，如 'sub-001'
        
    Returns:
        Tuple[np.ndarray, float, List[str]]: 
            - EEG数据 (n_channels, n_samples)
            - 采样率
            - 通道名列表
    """
    # 构建文件路径
    eeg_file = PREPROCESS_DIR / subject_id / "eeg" / f"{subject_id}_task-eyesclosed_eeg.set"
    
    if not eeg_file.exists():
        raise FileNotFoundError(f"EEG文件不存在: {eeg_file}")
    
    # 使用MNE加载.set文件
    raw = mne.io.read_raw_eeglab(str(eeg_file), preload=True, verbose=False)
    
    # 获取数据
    data = raw.get_data()  # shape: (n_channels, n_samples)
    sfreq = raw.info['sfreq']
    ch_names = raw.ch_names
    
    logger.debug(f"加载 {subject_id}: {data.shape}, 采样率={sfreq}Hz, 通道数={len(ch_names)}")
    
    return data, sfreq, ch_names


def load_all_subjects_data(
    groups: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    加载指定组别的所有被试数据
    
    Args:
        groups: 要加载的组别列表，如 ['A', 'C']。None表示加载所有
        
    Returns:
        Dict[str, Dict]: {subject_id: {'data': ndarray, 'sfreq': float, 
                                       'ch_names': list, 'group': str, 
                                       'age': int, 'gender': str, 'mmse': int}}
    """
    # 加载被试信息
    participants_df = load_participants_info()
    
    # 筛选指定组别
    if groups is not None:
        participants_df = participants_df[participants_df['Group'].isin(groups)]
        logger.info(f"筛选组别 {groups}: {len(participants_df)} 个被试")
    
    all_data = {}
    loaded_count = 0
    failed_count = 0
    
    for _, row in participants_df.iterrows():
        subject_id = row['participant_id']
        try:
            data, sfreq, ch_names = load_eeg_data(subject_id)
            
            all_data[subject_id] = {
                'data': data,
                'sfreq': sfreq,
                'ch_names': ch_names,
                'group': row['Group'],
                'age': row['Age'],
                'gender': row['Gender'],
                'mmse': row['MMSE']
            }
            loaded_count += 1
            
        except Exception as e:
            logger.error(f"加载 {subject_id} 失败: {e}")
            failed_count += 1
    
    logger.info(f"数据加载完成: 成功 {loaded_count}, 失败 {failed_count}")
    
    return all_data


def get_subjects_by_task(
    task: str, 
    participants_df: Optional[pd.DataFrame] = None
) -> Tuple[List[str], List[str]]:
    """
    根据分类任务获取被试ID列表
    
    Args:
        task: 任务名称 'AD_CN' 或 'FTD_CN'
        participants_df: 被试信息DataFrame，None则重新加载
        
    Returns:
        Tuple[List[str], List[str]]: (正类被试列表, 负类被试列表)
    """
    from config import CLASSIFICATION_TASKS
    
    if task not in CLASSIFICATION_TASKS:
        raise ValueError(f"未知任务: {task}. 可选: {list(CLASSIFICATION_TASKS.keys())}")
    
    task_config = CLASSIFICATION_TASKS[task]
    
    if participants_df is None:
        participants_df = load_participants_info()
    
    positive_subjects = participants_df[
        participants_df['Group'] == task_config['positive']
    ]['participant_id'].tolist()
    
    negative_subjects = participants_df[
        participants_df['Group'] == task_config['negative']
    ]['participant_id'].tolist()
    
    logger.info(f"任务 {task_config['name']}: "
                f"正类({task_config['positive']})={len(positive_subjects)}人, "
                f"负类({task_config['negative']})={len(negative_subjects)}人")
    
    return positive_subjects, negative_subjects


def describe_dataset():
    """
    打印数据集描述统计信息
    """
    df = load_participants_info()
    
    print("\n" + "="*60)
    print("数据集统计信息")
    print("="*60)
    
    print(f"\n总被试数: {len(df)}")
    
    # 各组统计
    print("\n各组统计:")
    for group in ['A', 'C', 'F']:
        group_df = df[df['Group'] == group]
        group_name = GROUP_LABELS[group]
        print(f"\n  {group_name} ({group}): {len(group_df)} 人")
        print(f"    年龄: {group_df['Age'].mean():.1f} ± {group_df['Age'].std():.1f} 岁")
        print(f"    MMSE: {group_df['MMSE'].mean():.2f} ± {group_df['MMSE'].std():.2f}")
        print(f"    性别: 男 {len(group_df[group_df['Gender']=='M'])}, "
              f"女 {len(group_df[group_df['Gender']=='F'])}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # 测试数据加载
    print("测试数据加载模块...")
    
    # 描述数据集
    describe_dataset()
    
    # 测试加载单个被试
    print("\n测试加载单个被试 (sub-001)...")
    data, sfreq, ch_names = load_eeg_data('sub-001')
    print(f"  数据形状: {data.shape}")
    print(f"  采样率: {sfreq} Hz")
    print(f"  通道: {ch_names}")
    print(f"  时长: {data.shape[1]/sfreq:.1f} 秒")
    
    # 测试获取任务被试
    print("\n测试获取分类任务被试...")
    for task in ['AD_CN', 'FTD_CN']:
        pos, neg = get_subjects_by_task(task)
        print(f"  {task}: 正类 {len(pos)} 人, 负类 {len(neg)} 人")

