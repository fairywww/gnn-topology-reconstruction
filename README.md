# gnn-topology-reconstruction

## 项目简介
本项目用于基于图神经网络（GNN）进行网络拓扑重建与分析，适用于网络科学、通信网络等相关领域的研究与实验。

## 主要功能
- 网络拓扑数据集处理与分析
- GNN模型（GCN、GAT、GraphSAGE）构建与训练
- 网络结构重建与性能评估（AUC、AP等指标）
- 结果自动保存与可视化

## 依赖环境
- Python 3.9+
- PyTorch
- torch-geometric
- networkx
- numpy（推荐1.26.x，避免2.x兼容性问题）
- pandas
- matplotlib
- 其它依赖详见 `requirements.txt`

## 快速开始
1. 克隆仓库：
   ```bash
   git clone https://github.com/fairywww/gnn-topology-reconstruction.git
   cd gnn-topology-reconstruction
   ```
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   # 如遇numpy兼容问题，建议：pip install numpy==1.26.4
   ```
3. 运行主实验脚本：
   ```bash
   python3 main.py
   ```
   - 实验将自动遍历数据集，训练GCN、GAT、SAGE三种模型，输出AUC/AP等指标。
   - 结果、配置和性能摘要会自动保存在 `results/` 目录下。

## 目录结构
- `src/`：核心源码（数据处理、模型、训练等）
- `results/`：实验结果、性能摘要、可视化图片
- `topologyzoo-main/`：网络拓扑数据集（.graphml等）
- `compare/`：对比实验脚本
- `docs/`：项目文档、分析总结
- `main.py`：主实验入口脚本
- `requirements.txt`：依赖说明

## 实验结果示例
- GCN: 平均AUC = 0.6922, 平均AP = 0.7788
- GAT: 平均AUC = 0.5501, 平均AP = 0.6882
- SAGE: 平均AUC = 0.7590, 平均AP = 0.8144
- 详细结果见 `results/gnn_results_*.csv`，性能摘要见 `results/performance_summary_*.json`

## 常见问题FAQ
- **Q: 运行时报 numpy 相关错误怎么办？**
  - A: 建议降级 numpy 到 1.26.x 版本：`pip install numpy==1.26.4`
- **Q: 依赖库太大/推送失败？**
  - A: 本项目已配置 `.gitignore`，无需上传虚拟环境和依赖包。
- **Q: 如何自定义实验？**
  - A: 可修改 `main.py` 或 `src/` 下相关脚本，支持自定义模型、数据等。

## 联系方式
如有问题或合作意向，请通过GitHub Issue联系作者。 