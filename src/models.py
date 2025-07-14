#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNN模型定义模块
包含GCN、GAT、GraphSAGE等图神经网络模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv


class GNNLinkPredictor(nn.Module):
    """GNN链接预测模型"""
    
    def __init__(self, in_channels, hidden_channels, out_channels, model_type='gcn'):
        super(GNNLinkPredictor, self).__init__()
        self.model_type = model_type
        
        # 编码器层
        if model_type == 'gcn':
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
        elif model_type == 'gat':
            # 优化GAT配置：使用更多注意力头，添加dropout
            self.conv1 = GATConv(in_channels, hidden_channels, heads=8, concat=False, dropout=0.2)
            self.conv2 = GATConv(hidden_channels, hidden_channels, heads=8, concat=False, dropout=0.2)
        elif model_type == 'sage':
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        
        # 预测层
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels // 2, 1)
        )
        
    def encode(self, x, edge_index):
        """编码节点特征"""
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)  # 添加额外的激活函数
        return x
    
    def decode(self, z, edge_label_index):
        """解码边标签"""
        row, col = edge_label_index
        z = torch.cat([z[row], z[col]], dim=-1)
        return self.predictor(z).view(-1)
    
    def forward(self, x, edge_index, edge_label_index):
        """前向传播"""
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


class AdvancedGNNLinkPredictor(nn.Module):
    """高级GNN链接预测模型 - 支持更多层和批归一化"""
    
    def __init__(self, in_channels, hidden_channels, out_channels, model_type='gcn', num_layers=3):
        super(AdvancedGNNLinkPredictor, self).__init__()
        self.model_type = model_type
        self.num_layers = num_layers
        
        # 编码器层
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                in_dim = in_channels
            else:
                in_dim = hidden_channels
                
            if model_type == 'gcn':
                self.convs.append(GCNConv(in_dim, hidden_channels))
            elif model_type == 'gat':
                # 优化GAT：使用更多注意力头，更好的dropout配置
                if i == 0:
                    self.convs.append(GATConv(in_dim, hidden_channels, heads=8, concat=False, dropout=0.3))
                elif i == self.num_layers - 1:
                    self.convs.append(GATConv(in_dim, hidden_channels, heads=1, concat=False, dropout=0.1))
                else:
                    self.convs.append(GATConv(in_dim, hidden_channels, heads=4, concat=False, dropout=0.2))
            elif model_type == 'sage':
                self.convs.append(SAGEConv(in_dim, hidden_channels))
            
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # 预测层
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels * 2),
            nn.BatchNorm1d(hidden_channels * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels // 2, 1)
        )
        
    def encode(self, x, edge_index):
        """编码节点特征"""
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)
        return x
    
    def decode(self, z, edge_label_index):
        """解码边标签"""
        row, col = edge_label_index
        z = torch.cat([z[row], z[col]], dim=-1)
        return self.predictor(z).view(-1)
    
    def forward(self, x, edge_index, edge_label_index):
        """前向传播"""
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index) 