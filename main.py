#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNN拓扑动物园链接预测主程序
使用模块化结构，支持配置管理和结果可视化
"""

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from src.models import GNNLinkPredictor, AdvancedGNNLinkPredictor
from src.dataset import TopologyZooDataset
from src.trainer import GNNTrainer, ExperimentRunner
from src.utils import ResultVisualizer, ResultSaver, ConfigManager


def main():
    """主函数"""
    print("=== GNN拓扑动物园链接预测实验 ===")
    
    # 加载配置
    config_manager = ConfigManager()
    config = config_manager.config
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化组件
    dataset = TopologyZooDataset(config['data']['data_dir'])
    trainer = GNNTrainer(device=str(device))
    experiment_runner = ExperimentRunner(trainer, device=str(device))
    visualizer = ResultVisualizer()
    saver = ResultSaver()
    
    # 加载数据集
    graphs = dataset.load_graphs(
        min_nodes=config['data']['min_nodes'],
        max_nodes=config['data']['max_nodes']
    )
    
    if len(graphs) == 0:
        print("没有找到符合条件的图！")
        return
    
    # 选择实验图（处理所有符合条件的图）
    selected_graphs = []
    for graph in graphs:
        if (config['experiment_setup']['min_nodes_per_graph'] <= graph.number_of_nodes() <= config['experiment_setup']['max_nodes_per_graph'] and 
            graph.number_of_edges() >= config['experiment_setup']['min_edges_per_graph']):
            selected_graphs.append(graph)
            if len(selected_graphs) >= config['experiment_setup']['max_graphs']:
                break
    
    print(f"选择了 {len(selected_graphs)} 个图进行实验")
    print(f"图大小范围: {min([g.number_of_nodes() for g in selected_graphs])} - {max([g.number_of_nodes() for g in selected_graphs])} 节点")
    print(f"边数范围: {min([g.number_of_edges() for g in selected_graphs])} - {max([g.number_of_edges() for g in selected_graphs])} 边")
    
    # 实验参数
    hidden_channels = config['model']['hidden_channels']
    lr = config['training']['learning_rate']
    epochs = config['training']['epochs']
    model_types = config['model']['model_types']
    
    # 结果存储
    results = {
        'train_losses': [],
        'val_aucs': [],
        'test_aucs': {},
        'test_aps': {},
        'graph_sizes': [g.number_of_nodes() for g in selected_graphs]
    }
    
    # 对每个图进行实验
    for i, graph in enumerate(selected_graphs):
        print(f"\n--- 实验图 {i+1}/{len(selected_graphs)}: {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边 ---")
        
        # 准备数据
        train_data, val_data, test_data = dataset.prepare_pyg_data(
            graph, 
            test_ratio=config['data']['test_ratio'],
            val_ratio=config['data']['val_ratio']
        )
        
        # 检查数据是否成功加载
        if train_data is None or val_data is None or test_data is None:
            print(f"跳过图 {i+1}，图太小或数据加载失败")
            continue
        
        # 创建模型和优化器
        models = {}
        optimizers = {}
        schedulers = {}
        
        for model_type in model_types:
            # 确保train_data不为None
            if train_data is not None and hasattr(train_data, 'x') and train_data.x is not None:
                in_channels = train_data.x.size(1)
            else:
                print(f"跳过模型 {model_type}，无法获取输入通道数")
                continue
                
            if config['model']['use_advanced_model']:
                model = AdvancedGNNLinkPredictor(
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    out_channels=hidden_channels,
                    model_type=model_type,
                    num_layers=config['model']['num_layers']
                ).to(device)
            else:
                model = GNNLinkPredictor(
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    out_channels=hidden_channels,
                    model_type=model_type
                ).to(device)
            
            optimizer = optim.Adam(
                model.parameters(), 
                lr=lr, 
                weight_decay=config['training']['weight_decay']
            )
            
            # 为GAT模型添加学习率调度器
            if model_type == 'gat' and config['training']['use_scheduler']:
                scheduler = lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='max', factor=0.5, patience=20, verbose=True
                )
            else:
                scheduler = None
            
            models[model_type] = model
            optimizers[model_type] = optimizer
            schedulers[model_type] = scheduler
        
        # 运行实验
        experiment_results = experiment_runner.run_multiple_experiments(
            models, train_data, val_data, test_data, optimizers, schedulers, epochs
        )
        
        # 保存结果
        for model_type in model_types:
            if model_type not in results['test_aucs']:
                results['test_aucs'][model_type] = []
                results['test_aps'][model_type] = []
            
            results['test_aucs'][model_type].append(experiment_results[model_type]['test_auc'])
            results['test_aps'][model_type].append(experiment_results[model_type]['test_ap'])
    
    # 计算平均性能
    print("\n=== 最终结果 ===")
    for model_type in model_types:
        avg_auc = np.mean(results['test_aucs'][model_type])
        avg_ap = np.mean(results['test_aps'][model_type])
        print(f"{model_type.upper()}: 平均AUC = {avg_auc:.4f}, 平均AP = {avg_ap:.4f}")
    
    # 可视化和保存结果
    visualizer.plot_training_curves(results, 'results/training_curves.png')
    visualizer.plot_model_comparison(results, 'results/model_comparison.png')
    
    saver.save_results_to_csv(results)
    saver.save_experiment_config(config)
    saver.save_model_performance(results)
    
    print("\n实验完成！结果已保存到 results/ 目录")


if __name__ == "__main__":
    main() 