#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练器模块
包含模型训练、验证和评估功能
"""

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np


class GNNTrainer:
    """GNN模型训练器"""
    
    def __init__(self, device='cpu'):
        self.device = device
        
    def train_model(self, model, train_data, val_data, optimizer, epochs=200, 
                   early_stopping_patience=50, verbose=True):
        """训练模型"""
        model.train()
        best_val_auc = 0
        patience_counter = 0
        train_losses = []
        val_aucs = []
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # 前向传播
            out = model(train_data.x.to(self.device), train_data.edge_index.to(self.device), 
                       train_data.edge_label_index.to(self.device))
            
            # 计算损失
            loss = F.binary_cross_entropy_with_logits(out, train_data.edge_label.to(self.device))
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 验证
            model.eval()
            with torch.no_grad():
                val_out = model(val_data.x.to(self.device), val_data.edge_index.to(self.device), 
                              val_data.edge_label_index.to(self.device))
                val_pred = torch.sigmoid(val_out)
                val_auc = roc_auc_score(val_data.edge_label.cpu().numpy(), val_pred.cpu().numpy())
            
            model.train()
            
            train_losses.append(loss.item())
            val_aucs.append(val_auc)
            
            # 早停检查
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"早停在第 {epoch} 轮，最佳验证AUC: {best_val_auc:.4f}")
                break
            
            if verbose and epoch % 50 == 0:
                print(f'Epoch {epoch:03d}: Loss: {loss:.4f}, Val AUC: {val_auc:.4f}')
        
        return train_losses, val_aucs, best_val_auc
    
    def evaluate_model(self, model, test_data):
        """评估模型"""
        model.eval()
        with torch.no_grad():
            test_out = model(test_data.x.to(self.device), test_data.edge_index.to(self.device), 
                           test_data.edge_label_index.to(self.device))
            test_pred = torch.sigmoid(test_out)
            
            test_auc = roc_auc_score(test_data.edge_label.cpu().numpy(), test_pred.cpu().numpy())
            test_ap = average_precision_score(test_data.edge_label.cpu().numpy(), test_pred.cpu().numpy())
        
        return test_auc, test_ap
    
    def train_with_scheduler(self, model, train_data, val_data, optimizer, scheduler, 
                           epochs=200, early_stopping_patience=50, verbose=True):
        """使用学习率调度器训练模型"""
        model.train()
        best_val_auc = 0
        patience_counter = 0
        train_losses = []
        val_aucs = []
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # 前向传播
            out = model(train_data.x.to(self.device), train_data.edge_index.to(self.device), 
                       train_data.edge_label_index.to(self.device))
            
            # 计算损失
            loss = F.binary_cross_entropy_with_logits(out, train_data.edge_label.to(self.device))
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 验证
            model.eval()
            with torch.no_grad():
                val_out = model(val_data.x.to(self.device), val_data.edge_index.to(self.device), 
                              val_data.edge_label_index.to(self.device))
                val_pred = torch.sigmoid(val_out)
                val_auc = roc_auc_score(val_data.edge_label.cpu().numpy(), val_pred.cpu().numpy())
            model.train()
            
            # 更新学习率（放到val_auc赋值后）
            scheduler.step(val_auc)
            
            train_losses.append(loss.item())
            val_aucs.append(val_auc)
            
            # 早停检查
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"早停在第 {epoch} 轮，最佳验证AUC: {best_val_auc:.4f}")
                break
            
            if verbose and epoch % 50 == 0:
                print(f'Epoch {epoch:03d}: Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
        return train_losses, val_aucs, best_val_auc


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, trainer, device='cpu'):
        self.trainer = trainer
        self.device = device
        
    def run_single_experiment(self, model, train_data, val_data, test_data, 
                            optimizer, epochs=200, model_name="Model"):
        """运行单个实验"""
        print(f"\n测试模型: {model_name}")
        
        # 训练模型
        train_losses, val_aucs, best_val_auc = self.trainer.train_model(
            model, train_data, val_data, optimizer, epochs
        )
        
        # 评估模型
        test_auc, test_ap = self.trainer.evaluate_model(model, test_data)
        
        print(f"测试结果 - AUC: {test_auc:.4f}, AP: {test_ap:.4f}")
        
        return {
            'train_losses': train_losses,
            'val_aucs': val_aucs,
            'best_val_auc': best_val_auc,
            'test_auc': test_auc,
            'test_ap': test_ap
        }
    
    def run_multiple_experiments(self, models, train_data, val_data, test_data, 
                               optimizers, schedulers=None, epochs=200):
        """运行多个模型实验"""
        results = {}
        
        if schedulers is None:
            schedulers = {name: None for name in models.keys()}
        
        for model_name in models.keys():
            model = models[model_name]
            optimizer = optimizers[model_name]
            scheduler = schedulers.get(model_name)
            
            if scheduler is not None:
                # 使用带调度器的训练
                result = self.run_single_experiment_with_scheduler(
                    model, train_data, val_data, test_data, optimizer, scheduler, epochs, model_name
                )
            else:
                # 使用普通训练
                result = self.run_single_experiment(
                    model, train_data, val_data, test_data, optimizer, epochs, model_name
                )
            results[model_name] = result
            
        return results
    
    def run_single_experiment_with_scheduler(self, model, train_data, val_data, test_data, 
                                           optimizer, scheduler, epochs=200, model_name="Model"):
        """运行带学习率调度器的单个实验"""
        print(f"\n测试模型: {model_name} (带学习率调度器)")
        
        # 训练模型
        train_losses, val_aucs, best_val_auc = self.trainer.train_with_scheduler(
            model, train_data, val_data, optimizer, scheduler, epochs
        )
        
        # 评估模型
        test_auc, test_ap = self.trainer.evaluate_model(model, test_data)
        
        print(f"测试结果 - AUC: {test_auc:.4f}, AP: {test_ap:.4f}")
        
        return {
            'train_losses': train_losses,
            'val_aucs': val_aucs,
            'best_val_auc': best_val_auc,
            'test_auc': test_auc,
            'test_ap': test_ap
        } 