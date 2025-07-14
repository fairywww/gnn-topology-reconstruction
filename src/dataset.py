#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集处理模块
包含拓扑动物园数据集的加载、预处理和特征工程
"""

import os
import glob
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling


class TopologyZooDataset:
    """拓扑动物园数据集处理类"""
    
    def __init__(self, data_dir='topologyzoo-main/graphml'):
        self.data_dir = data_dir
        self.graphs = []
        
    def load_graphs(self, min_nodes=5, max_nodes=200):
        """加载符合条件的图"""
        graph_files = glob.glob(os.path.join(self.data_dir, '*.graphml'))
        
        for file_path in graph_files:
            try:
                G = nx.read_graphml(file_path)
                
                # 过滤条件
                if G.number_of_nodes() < min_nodes or G.number_of_nodes() > max_nodes:
                    continue
                    
                # 确保图是连通的
                if not nx.is_connected(G):
                    largest_cc = max(nx.connected_components(G), key=len)
                    G = G.subgraph(largest_cc).copy()
                
                # 重新编号节点
                G = nx.convert_node_labels_to_integers(G)
                
                self.graphs.append(G)
                print(f"加载图: {os.path.basename(file_path)} - 节点: {G.number_of_nodes()}, 边: {G.number_of_edges()}")
                
            except Exception as e:
                print(f"加载图 {file_path} 时出错: {e}")
                continue
        
        print(f"总共加载了 {len(self.graphs)} 个图")
        return self.graphs
    
    def extract_node_features(self, graph):
        """提取节点特征"""
        node_features = []
        for node in graph.nodes():
            features = []
            
            # 1. 地理位置特征（归一化到[0,1]）
            if 'Latitude' in graph.nodes[node]:
                lat = float(graph.nodes[node]['Latitude'])
                lat_norm = (lat + 90) / 180
                features.append(lat_norm)
            else:
                features.append(0.5)
                
            if 'Longitude' in graph.nodes[node]:
                lon = float(graph.nodes[node]['Longitude'])
                lon_norm = (lon + 180) / 360
                features.append(lon_norm)
            else:
                features.append(0.5)
            
            # 2. 节点类型特征（one-hot编码）
            node_type = graph.nodes[node].get('type', 'Unknown')
            type_mapping = {'PoP': 0, 'Cityring': 1, 'Unknown': 2}
            type_id = type_mapping.get(node_type, 2)
            type_features = [0, 0, 0]
            type_features[type_id] = 1
            features.extend(type_features)
            
            # 3. 内部节点标志
            internal = int(graph.nodes[node].get('Internal', 0))
            features.append(internal)
            
            # 4. 度数特征（归一化）
            degree = graph.degree(node)
            max_degree = max(graph.degree(n) for n in graph.nodes())
            degree_norm = degree / max_degree if max_degree > 0 else 0
            features.append(degree_norm)
            
            # 5. 中心性特征
            degree_cent = nx.degree_centrality(graph)[node]
            closeness_cent = nx.closeness_centrality(graph)[node]
            betweenness_cent = nx.betweenness_centrality(graph)[node]
            features.extend([degree_cent, closeness_cent, betweenness_cent])
            
            node_features.append(features)
        
        return torch.tensor(node_features, dtype=torch.float)
    
    def prepare_pyg_data(self, graph, test_ratio=0.1, val_ratio=0.1):
        """将NetworkX图转换为PyTorch Geometric数据格式"""
        # 提取节点特征
        x = self.extract_node_features(graph)
        print(f"[特征信息] 节点特征维度: {x.shape}, 特征包括: 经纬度(2) + 类型(3) + 内部标志(1) + 度数(1) + 中心性(3)")
        
        # 边索引
        edge_index = torch.tensor(list(graph.edges()), dtype=torch.long).t().contiguous()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # 添加反向边
        
        # 分割边为训练/验证/测试集
        num_edges = edge_index.size(1) // 2
        
        # 确保每个集合至少有一个样本
        if num_edges < 10:
            # 对于小图，使用更简单的分割策略
            if num_edges >= 3:
                train_size = num_edges - 2
                val_size = 1
                test_size = 1
            else:
                # 如果边太少，跳过这个图
                print(f"[警告] 图太小 ({num_edges} 条边)，跳过")
                return None, None, None
        else:
            test_size = max(1, int(num_edges * test_ratio))
            val_size = max(1, int(num_edges * val_ratio))
            train_size = num_edges - test_size - val_size
        
        indices = torch.randperm(num_edges)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # 创建训练数据
        train_edges = edge_index[:, train_indices]
        train_neg_edges = negative_sampling(train_edges, num_nodes=graph.number_of_nodes(), 
                                          num_neg_samples=train_edges.size(1))
        
        # 检查负采样是否采到正样本
        train_pos_set = set(map(tuple, train_edges.t().cpu().numpy()))
        train_neg_set = set(map(tuple, train_neg_edges.t().cpu().numpy()))
        train_overlap = train_pos_set & train_neg_set
        print(f"[调试] 训练集：正样本 {len(train_pos_set)}，负样本 {len(train_neg_set)}，负采样重叠 {len(train_overlap)}")
        
        train_edge_label_index = torch.cat([train_edges, train_neg_edges], dim=1)
        train_edge_label = torch.cat([torch.ones(train_edges.size(1)), torch.zeros(train_neg_edges.size(1))])
        
        # 创建验证数据
        val_edges = edge_index[:, val_indices]
        val_neg_edges = negative_sampling(val_edges, num_nodes=graph.number_of_nodes(), 
                                        num_neg_samples=val_edges.size(1))
        val_pos_set = set(map(tuple, val_edges.t().cpu().numpy()))
        val_neg_set = set(map(tuple, val_neg_edges.t().cpu().numpy()))
        val_overlap = val_pos_set & val_neg_set
        print(f"[调试] 验证集：正样本 {len(val_pos_set)}，负样本 {len(val_neg_set)}，负采样重叠 {len(val_overlap)}")
        
        val_edge_label_index = torch.cat([val_edges, val_neg_edges], dim=1)
        val_edge_label = torch.cat([torch.ones(val_edges.size(1)), torch.zeros(val_neg_edges.size(1))])
        
        # 创建测试数据
        test_edges = edge_index[:, test_indices]
        test_neg_edges = negative_sampling(test_edges, num_nodes=graph.number_of_nodes(), 
                                         num_neg_samples=test_edges.size(1))
        test_pos_set = set(map(tuple, test_edges.t().cpu().numpy()))
        test_neg_set = set(map(tuple, test_neg_edges.t().cpu().numpy()))
        test_overlap = test_pos_set & test_neg_set
        print(f"[调试] 测试集：正样本 {len(test_pos_set)}，负样本 {len(test_neg_set)}，负采样重叠 {len(test_overlap)}")
        
        test_edge_label_index = torch.cat([test_edges, test_neg_edges], dim=1)
        test_edge_label = torch.cat([torch.ones(test_edges.size(1)), torch.zeros(test_neg_edges.size(1))])
        
        # 创建PyG数据对象 - 使用完整图结构进行训练
        full_edge_index = edge_index  # 使用完整图的边
        
        train_data = Data(x=x, edge_index=full_edge_index, edge_label_index=train_edge_label_index, 
                         edge_label=train_edge_label)
        val_data = Data(x=x, edge_index=full_edge_index, edge_label_index=val_edge_label_index, 
                       edge_label=val_edge_label)
        test_data = Data(x=x, edge_index=full_edge_index, edge_label_index=test_edge_label_index, 
                        edge_label=test_edge_label)
        
        return train_data, val_data, test_data
    
    def select_graphs_by_size(self, min_nodes=50, max_nodes=100, min_edges=60, max_graphs=5):
        """根据图大小选择特定的图"""
        selected_graphs = []
        for graph in self.graphs:
            if (min_nodes <= graph.number_of_nodes() <= max_nodes and 
                graph.number_of_edges() >= min_edges):
                selected_graphs.append(graph)
                if len(selected_graphs) >= max_graphs:
                    break
        
        if len(selected_graphs) == 0:
            # 如果没有找到符合条件的图，使用前5个较大的图
            selected_graphs = sorted(self.graphs, key=lambda g: g.number_of_nodes(), reverse=True)[:max_graphs]
        
        return selected_graphs 