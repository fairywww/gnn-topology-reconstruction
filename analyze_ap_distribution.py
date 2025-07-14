#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析不同图的AP分布情况
找出影响AP表现的因素
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os

def analyze_ap_distribution():
    """分析AP分布情况"""
    
    # 读取最新的结果文件
    results_dir = 'results'
    csv_files = [f for f in os.listdir(results_dir) if f.startswith('gnn_results_') and f.endswith('.csv')]
    latest_file = sorted(csv_files)[-1]
    file_path = os.path.join(results_dir, latest_file)
    
    print(f"分析文件: {file_path}")
    
    # 读取数据
    df = pd.read_csv(file_path)
    
    # 按模型分组分析
    models = df['Model'].unique()
    
    print("\n=== AP分布分析 ===")
    
    for model in models:
        model_data = df[df['Model'] == model]
        ap_values = model_data['AP'].values
        
        print(f"\n{model.upper()}模型:")
        print(f"  样本数量: {len(ap_values)}")
        print(f"  平均AP: {np.mean(ap_values):.4f}")
        print(f"  标准差: {np.std(ap_values):.4f}")
        print(f"  最小值: {np.min(ap_values):.4f}")
        print(f"  最大值: {np.max(ap_values):.4f}")
        print(f"  中位数: {np.median(ap_values):.4f}")
        
        # AP分布统计
        perfect_ap = np.sum(ap_values == 1.0)
        high_ap = np.sum(ap_values >= 0.8)
        medium_ap = np.sum((ap_values >= 0.6) & (ap_values < 0.8))
        low_ap = np.sum((ap_values >= 0.4) & (ap_values < 0.6))
        very_low_ap = np.sum(ap_values < 0.4)
        
        print(f"  AP=1.0: {perfect_ap} ({perfect_ap/len(ap_values)*100:.1f}%)")
        print(f"  AP≥0.8: {high_ap} ({high_ap/len(ap_values)*100:.1f}%)")
        print(f"  0.6≤AP<0.8: {medium_ap} ({medium_ap/len(ap_values)*100:.1f}%)")
        print(f"  0.4≤AP<0.6: {low_ap} ({low_ap/len(ap_values)*100:.1f}%)")
        print(f"  AP<0.4: {very_low_ap} ({very_low_ap/len(ap_values)*100:.1f}%)")
    
    # 找出表现最好和最差的图
    print("\n=== 表现分析 ===")
    
    # 计算每个图（每3行）的平均AP
    graph_performance = []
    for i in range(0, len(df), 3):
        if i + 2 < len(df):
            graph_aps = df.iloc[i:i+3]['AP'].values
            graph_models = df.iloc[i:i+3]['Model'].values
            avg_ap = np.mean(graph_aps)
            graph_performance.append({
                'graph_id': i // 3 + 1,
                'avg_ap': avg_ap,
                'gcn_ap': graph_aps[0] if graph_models[0] == 'GCN' else graph_aps[1] if graph_models[1] == 'GCN' else graph_aps[2],
                'gat_ap': graph_aps[0] if graph_models[0] == 'GAT' else graph_aps[1] if graph_models[1] == 'GAT' else graph_aps[2],
                'sage_ap': graph_aps[0] if graph_models[0] == 'SAGE' else graph_aps[1] if graph_models[1] == 'SAGE' else graph_aps[2],
                'aps': graph_aps
            })
    
    # 按平均AP排序
    graph_performance.sort(key=lambda x: x['avg_ap'], reverse=True)
    
    print("\n表现最好的10个图:")
    for i, graph in enumerate(graph_performance[:10]):
        print(f"  图{graph['graph_id']}: 平均AP={graph['avg_ap']:.4f} (GCN:{graph['gcn_ap']:.3f}, GAT:{graph['gat_ap']:.3f}, SAGE:{graph['sage_ap']:.3f})")
    
    print("\n表现最差的10个图:")
    for i, graph in enumerate(graph_performance[-10:]):
        print(f"  图{graph['graph_id']}: 平均AP={graph['avg_ap']:.4f} (GCN:{graph['gcn_ap']:.3f}, GAT:{graph['gat_ap']:.3f}, SAGE:{graph['sage_ap']:.3f})")
    
    # 分析AP差异的原因
    print("\n=== AP差异原因分析 ===")
    
    # 计算每个图的AP方差
    ap_variances = []
    for graph in graph_performance:
        variance = np.var(graph['aps'])
        ap_variances.append({
            'graph_id': graph['graph_id'],
            'avg_ap': graph['avg_ap'],
            'variance': variance,
            'max_diff': np.max(graph['aps']) - np.min(graph['aps'])
        })
    
    # 找出模型间差异最大的图
    ap_variances.sort(key=lambda x: x['variance'], reverse=True)
    
    print("\n模型间AP差异最大的10个图:")
    for i, graph in enumerate(ap_variances[:10]):
        print(f"  图{graph['graph_id']}: 方差={graph['variance']:.4f}, 最大差异={graph['max_diff']:.4f}, 平均AP={graph['avg_ap']:.4f}")
    
    # 分析AP与AUC的关系
    print("\n=== AP与AUC关系分析 ===")
    
    for model in models:
        model_data = df[df['Model'] == model]
        correlation = np.corrcoef(model_data['AUC'], model_data['AP'])[0, 1]
        print(f"{model.upper()}: AP与AUC相关系数 = {correlation:.4f}")
    
    # 创建可视化
    create_visualizations(df, graph_performance)
    
    return df, graph_performance

def create_visualizations(df, graph_performance):
    """创建可视化图表"""
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. AP分布直方图
    models = df['Model'].unique()
    colors = ['blue', 'red', 'green']
    
    for i, model in enumerate(models):
        model_data = df[df['Model'] == model]
        axes[0, 0].hist(model_data['AP'], bins=20, alpha=0.7, label=model.upper(), color=colors[i])
    
    axes[0, 0].set_xlabel('AP值')
    axes[0, 0].set_ylabel('频次')
    axes[0, 0].set_title('不同模型的AP分布')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 箱线图
    ap_data = [df[df['Model'] == model]['AP'].values for model in models]
    axes[0, 1].boxplot(ap_data, labels=[m.upper() for m in models])
    axes[0, 1].set_ylabel('AP值')
    axes[0, 1].set_title('AP分布箱线图')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. AP与AUC散点图
    for i, model in enumerate(models):
        model_data = df[df['Model'] == model]
        axes[1, 0].scatter(model_data['AUC'], model_data['AP'], alpha=0.6, label=model.upper(), color=colors[i])
    
    axes[1, 0].set_xlabel('AUC值')
    axes[1, 0].set_ylabel('AP值')
    axes[1, 0].set_title('AP与AUC关系')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 图性能分布
    avg_aps = [g['avg_ap'] for g in graph_performance]
    axes[1, 1].hist(avg_aps, bins=20, alpha=0.7, color='purple')
    axes[1, 1].set_xlabel('平均AP值')
    axes[1, 1].set_ylabel('图数量')
    axes[1, 1].set_title('图的平均AP分布')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/ap_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n可视化图表已保存为 results/ap_analysis.png")

if __name__ == "__main__":
    analyze_ap_distribution() 