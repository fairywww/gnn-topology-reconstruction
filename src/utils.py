#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具模块
包含可视化、结果保存、配置管理等功能
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class ResultVisualizer:
    """结果可视化工具"""
    
    def __init__(self, save_dir='results'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
    
    def plot_training_curves(self, results, save_path=None):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 训练损失
        axes[0, 0].plot(results['train_losses'])
        axes[0, 0].set_title('训练损失')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        
        # 验证AUC
        axes[0, 1].plot(results['val_aucs'])
        axes[0, 1].set_title('验证AUC')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC')
        
        # 模型性能比较
        models = list(results['test_aucs'].keys())
        aucs = [np.mean(results['test_aucs'][m]) for m in models]
        aps = [np.mean(results['test_aps'][m]) for m in models]

        x = np.arange(len(models))
        width = 0.35

        axes[1, 0].bar(x - width/2, aucs, width, label='AUC', alpha=0.8)
        axes[1, 0].bar(x + width/2, aps, width, label='AP', alpha=0.8)
        axes[1, 0].set_title('模型性能比较')
        axes[1, 0].set_xlabel('模型')
        axes[1, 0].set_ylabel('分数')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(models)
        axes[1, 0].legend()
        
        # 数据集统计
        if 'graph_sizes' in results:
            graph_sizes = results['graph_sizes']
            axes[1, 1].hist(graph_sizes, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('图大小分布')
            axes[1, 1].set_xlabel('节点数')
            axes[1, 1].set_ylabel('频次')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, results, save_path=None):
        """绘制模型比较图"""
        models = list(results['test_aucs'].keys())
        aucs = [np.mean(results['test_aucs'][m]) for m in models]
        aps = [np.mean(results['test_aps'][m]) for m in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # AUC比较
        bars1 = ax1.bar(models, aucs, alpha=0.8, color='skyblue')
        ax1.set_title('模型AUC比较')
        ax1.set_ylabel('AUC')
        ax1.set_ylim(0, 1)
        
        # 添加数值标签
        for bar, auc in zip(bars1, aucs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{auc:.3f}', ha='center', va='bottom')
        
        # AP比较
        bars2 = ax2.bar(models, aps, alpha=0.8, color='lightcoral')
        ax2.set_title('模型AP比较')
        ax2.set_ylabel('AP')
        ax2.set_ylim(0, 1)
        
        # 添加数值标签
        for bar, ap in zip(bars2, aps):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{ap:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class ResultSaver:
    """结果保存工具"""
    
    def __init__(self, save_dir='results'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def save_results_to_csv(self, results, filename=None):
        """保存结果到CSV文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gnn_results_{timestamp}.csv"
        
        filepath = os.path.join(self.save_dir, filename)
        
        # 创建结果DataFrame
        results_df = pd.DataFrame({
            'Model': [],
            'AUC': [],
            'AP': []
        })
        
        for model_type in results['test_aucs'].keys():
            for auc, ap in zip(results['test_aucs'][model_type], results['test_aps'][model_type]):
                results_df = pd.concat([results_df, pd.DataFrame({
                    'Model': [model_type.upper()],
                    'AUC': [auc],
                    'AP': [ap]
                })], ignore_index=True)
        
        results_df.to_csv(filepath, index=False)
        print(f"结果已保存到 {filepath}")
        return filepath
    
    def save_experiment_config(self, config, filename=None):
        """保存实验配置"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment_config_{timestamp}.json"
        
        filepath = os.path.join(self.save_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"配置已保存到 {filepath}")
        return filepath
    
    def save_model_performance(self, results, filename=None):
        """保存模型性能摘要"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_summary_{timestamp}.json"
        
        filepath = os.path.join(self.save_dir, filename)
        
        performance_summary = {}
        for model_type in results['test_aucs'].keys():
            avg_auc = np.mean(results['test_aucs'][model_type])
            avg_ap = np.mean(results['test_aps'][model_type])
            std_auc = np.std(results['test_aucs'][model_type])
            std_ap = np.std(results['test_aps'][model_type])
            
            performance_summary[model_type] = {
                'avg_auc': float(avg_auc),
                'avg_ap': float(avg_ap),
                'std_auc': float(std_auc),
                'std_ap': float(std_ap),
                'num_experiments': len(results['test_aucs'][model_type])
            }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(performance_summary, f, indent=2, ensure_ascii=False)
        
        print(f"性能摘要已保存到 {filepath}")
        return filepath


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file='config.json'):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self):
        """加载配置"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return self.get_default_config()
    
    def get_default_config(self):
        """获取默认配置"""
        return {
            'experiment': {
                'name': 'GNN_Topology_Zoo_Link_Prediction',
                'description': '使用GNN进行拓扑动物园数据集链接预测',
                'version': '1.0.0'
            },
            'data': {
                'data_dir': 'topologyzoo-main/graphml',
                'min_nodes': 10,
                'max_nodes': 150,
                'test_ratio': 0.1,
                'val_ratio': 0.1
            },
            'model': {
                'hidden_channels': 64,
                'model_types': ['gcn', 'gat', 'sage'],
                'use_advanced_model': False,
                'num_layers': 2
            },
            'training': {
                'epochs': 200,
                'learning_rate': 0.005,
                'weight_decay': 5e-4,
                'early_stopping_patience': 50,
                'use_scheduler': False
            },
            'experiment_setup': {
                'max_graphs': 5,
                'min_nodes_per_graph': 50,
                'max_nodes_per_graph': 100,
                'min_edges_per_graph': 60
            }
        }
    
    def save_config(self):
        """保存配置"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def update_config(self, updates):
        """更新配置"""
        def update_nested_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_nested_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self.config = update_nested_dict(self.config, updates)
        self.save_config()
    
    def get(self, key, default=None):
        """获取配置值"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value 