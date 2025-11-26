# -*- coding: utf-8 -*-
"""
特征提取模块 - 提取相对频带功率(RBP)特征
论文方法: 4秒窗口, 50%重叠, Welch PSD, 5频带RBP
"""

import numpy as np
from scipy import signal
from typing import Dict, List, Tuple, Optional
import pickle
import logging
from pathlib import Path
from tqdm import tqdm

from config import (
    FREQ_BANDS,
    WINDOW_SIZE,
    OVERLAP_RATIO,
    SAMPLING_RATE,
    FILTER_LOW,
    FILTER_HIGH,
    WELCH_NPERSEG,
    CACHE_DIR
)

logger = logging.getLogger(__name__)


def compute_psd_welch(
    data: np.ndarray, 
    sfreq: float,
    nperseg: int = WELCH_NPERSEG
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用Welch方法计算功率谱密度
    
    Args:
        data: EEG数据 (n_channels, n_samples) 或 (n_samples,)
        sfreq: 采样率
        nperseg: 每段长度
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (频率数组, PSD数组)
    """
    # 确保是2D
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    # 计算PSD
    freqs, psd = signal.welch(data, fs=sfreq, nperseg=nperseg)
    
    return freqs, psd


def compute_band_power(
    freqs: np.ndarray, 
    psd: np.ndarray, 
    band: Tuple[float, float]
) -> np.ndarray:
    """
    计算指定频带的功率
    
    Args:
        freqs: 频率数组
        psd: PSD数组 (n_channels, n_freqs)
        band: 频带范围 (low, high)
        
    Returns:
        np.ndarray: 各通道在该频带的功率
    """
    low, high = band
    idx = np.logical_and(freqs >= low, freqs <= high)
    
    # 使用梯形积分计算频带功率
    band_power = np.trapz(psd[:, idx], freqs[idx], axis=-1)
    
    return band_power


def compute_relative_band_power(
    data: np.ndarray,
    sfreq: float,
    freq_bands: Dict[str, Tuple[float, float]] = FREQ_BANDS,
    total_band: Tuple[float, float] = (FILTER_LOW, FILTER_HIGH)
) -> Dict[str, np.ndarray]:
    """
    计算各频带的相对功率(RBP)
    
    Args:
        data: EEG数据 (n_channels, n_samples)
        sfreq: 采样率
        freq_bands: 频带定义字典
        total_band: 总频率范围
        
    Returns:
        Dict[str, np.ndarray]: 各频带的相对功率 {band_name: power_array}
    """
    # 计算PSD
    freqs, psd = compute_psd_welch(data, sfreq)
    
    # 计算总功率
    total_power = compute_band_power(freqs, psd, total_band)
    
    # 避免除零
    total_power = np.maximum(total_power, 1e-10)
    
    # 计算各频带相对功率
    rbp = {}
    for band_name, band_range in freq_bands.items():
        band_power = compute_band_power(freqs, psd, band_range)
        rbp[band_name] = band_power / total_power
    
    return rbp


def segment_signal(
    data: np.ndarray,
    sfreq: float,
    window_size: float = WINDOW_SIZE,
    overlap_ratio: float = OVERLAP_RATIO
) -> List[np.ndarray]:
    """
    将信号分割成重叠的时间窗口
    
    Args:
        data: EEG数据 (n_channels, n_samples)
        sfreq: 采样率
        window_size: 窗口大小(秒)
        overlap_ratio: 重叠比例
        
    Returns:
        List[np.ndarray]: 分割后的epoch列表
    """
    n_channels, n_samples = data.shape
    
    # 计算样本数
    window_samples = int(window_size * sfreq)
    step_samples = int(window_samples * (1 - overlap_ratio))
    
    # 分割
    epochs = []
    start = 0
    while start + window_samples <= n_samples:
        epoch = data[:, start:start + window_samples]
        epochs.append(epoch)
        start += step_samples
    
    return epochs


def extract_features_single_subject(
    data: np.ndarray,
    sfreq: float,
    window_size: float = WINDOW_SIZE,
    overlap_ratio: float = OVERLAP_RATIO,
    aggregate_channels: bool = True,
    feature_order: str = 'by_channel'  # 'by_channel' 或 'by_band'
) -> np.ndarray:
    """
    提取单个被试的特征
    
    Args:
        data: EEG数据 (n_channels, n_samples)
        sfreq: 采样率
        window_size: 窗口大小(秒)
        overlap_ratio: 重叠比例
        aggregate_channels: 是否对通道取平均
        feature_order: 特征排列方式
            - 'by_channel': [ch1的5频带, ch2的5频带, ...] 共95个
            - 'by_band': [所有通道的delta, 所有通道的theta, ...] 共95个
        
    Returns:
        np.ndarray: 特征矩阵 (n_epochs, n_features)
                    如果aggregate_channels=True, n_features=5
                    否则 n_features=5*n_channels
    """
    # 分割信号
    epochs = segment_signal(data, sfreq, window_size, overlap_ratio)
    
    if len(epochs) == 0:
        logger.warning("没有足够的数据进行分割")
        return np.array([])
    
    # 提取每个epoch的特征
    all_features = []
    band_names = list(FREQ_BANDS.keys())
    n_channels = data.shape[0]
    
    for epoch in epochs:
        # 计算相对频带功率
        rbp = compute_relative_band_power(epoch, sfreq)
        
        if aggregate_channels:
            # 对所有通道取平均，得到5个特征
            features = np.array([rbp[band].mean() for band in band_names])
        else:
            if feature_order == 'by_channel':
                # 按通道排列: [ch1的5频带, ch2的5频带, ...]
                features = []
                for ch in range(n_channels):
                    for band in band_names:
                        features.append(rbp[band][ch])
                features = np.array(features)
            else:  # 'by_band'
                # 按频带排列: [所有通道的delta, 所有通道的theta, ...]
                features = np.concatenate([rbp[band] for band in band_names])
        
        all_features.append(features)
    
    return np.array(all_features)


def extract_features_all_subjects(
    subjects_data: Dict[str, Dict],
    window_size: float = WINDOW_SIZE,
    overlap_ratio: float = OVERLAP_RATIO,
    aggregate_channels: bool = True,
    feature_order: str = 'by_channel',
    use_cache: bool = True,
    cache_name: str = "features"
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """
    提取所有被试的特征
    
    Args:
        subjects_data: 所有被试的数据字典
        window_size: 窗口大小(秒)
        overlap_ratio: 重叠比例
        aggregate_channels: 是否对通道取平均
        feature_order: 特征排列方式 ('by_channel' 或 'by_band')
        use_cache: 是否使用缓存
        cache_name: 缓存文件名
        
    Returns:
        Tuple[Dict, Dict]: 
            - {subject_id: features_array}
            - {subject_id: label} (0 或 1)
    """
    cache_file = CACHE_DIR / f"{cache_name}.pkl"
    
    # 尝试加载缓存
    if use_cache and cache_file.exists():
        logger.info(f"从缓存加载特征: {cache_file}")
        with open(cache_file, 'rb') as f:
            cached = pickle.load(f)
        return cached['features'], cached['labels']
    
    logger.info("开始提取特征...")
    
    features_dict = {}
    labels_dict = {}
    
    for subject_id, subject_info in tqdm(subjects_data.items(), desc="提取特征"):
        data = subject_info['data']
        sfreq = subject_info['sfreq']
        group = subject_info['group']
        
        # 提取特征
        features = extract_features_single_subject(
            data, sfreq, window_size, overlap_ratio, aggregate_channels, feature_order
        )
        
        if len(features) > 0:
            features_dict[subject_id] = features
            labels_dict[subject_id] = group
            
            logger.debug(f"{subject_id}: {len(features)} epochs, {features.shape[1]} features")
    
    # 保存缓存
    if use_cache:
        with open(cache_file, 'wb') as f:
            pickle.dump({'features': features_dict, 'labels': labels_dict}, f)
        logger.info(f"特征已缓存至: {cache_file}")
    
    return features_dict, labels_dict


def prepare_data_for_task(
    features_dict: Dict[str, np.ndarray],
    labels_dict: Dict[str, str],
    positive_label: str,
    negative_label: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    为特定分类任务准备数据
    
    Args:
        features_dict: 特征字典
        labels_dict: 标签字典
        positive_label: 正类标签 (如 'A' for AD)
        negative_label: 负类标签 (如 'C' for CN)
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - X: 特征矩阵 (n_samples, n_features)
            - y: 标签数组 (n_samples,), 1为正类, 0为负类
            - subject_ids: 每个样本对应的被试ID
    """
    X_list = []
    y_list = []
    subject_ids_list = []
    
    for subject_id, features in features_dict.items():
        label = labels_dict[subject_id]
        
        if label == positive_label:
            y = 1
        elif label == negative_label:
            y = 0
        else:
            continue  # 跳过不相关的组
        
        n_epochs = len(features)
        X_list.append(features)
        y_list.extend([y] * n_epochs)
        subject_ids_list.extend([subject_id] * n_epochs)
    
    X = np.vstack(X_list)
    y = np.array(y_list)
    subject_ids = np.array(subject_ids_list)
    
    logger.info(f"准备数据: {len(X)} 个样本, "
                f"正类(1)={np.sum(y==1)}, 负类(0)={np.sum(y==0)}")
    
    return X, y, subject_ids


if __name__ == "__main__":
    # 测试特征提取
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    from data_loader import load_eeg_data, load_all_subjects_data
    
    print("测试特征提取模块...")
    
    # 测试单个被试
    print("\n1. 测试单个被试特征提取...")
    data, sfreq, ch_names = load_eeg_data('sub-001')
    features = extract_features_single_subject(data, sfreq)
    print(f"   输入数据: {data.shape}")
    print(f"   输出特征: {features.shape}")
    print(f"   每个epoch的特征: {list(FREQ_BANDS.keys())}")
    print(f"   特征示例 (第一个epoch): {features[0]}")
    
    # 测试PSD计算
    print("\n2. 测试PSD计算...")
    epochs = segment_signal(data, sfreq)
    print(f"   分割epoch数: {len(epochs)}")
    
    rbp = compute_relative_band_power(epochs[0], sfreq)
    print(f"   相对频带功率:")
    for band, power in rbp.items():
        print(f"     {band}: mean={power.mean():.4f}, std={power.std():.4f}")

