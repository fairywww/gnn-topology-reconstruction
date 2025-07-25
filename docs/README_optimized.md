# GNN拓扑还原与链路预测项目

## 项目简介

本项目基于图神经网络（GNN），针对真实运营商网络拓扑（Topology Zoo数据集）实现自动化链路预测与拓扑还原。相比传统方法，显著提升了预测精度和自动化水平，为网络智能运维和规划提供了高效工具。

## 项目亮点
- **真实数据驱动**：采用236个真实运营商网络拓扑，结果具备行业代表性
- **多模型对比**：支持GCN、GAT、GraphSAGE等主流GNN架构
- **自动特征工程**：融合地理位置、节点类型、中心性等多维特征
- **高性能链路预测**：AP最高可达0.90，显著优于传统方法
- **模块化设计**：易于扩展、维护和集成

## 快速上手

1. **环境准备**
```bash
python -m venv gnn_topology
source gnn_topology/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

2. **运行实验**
```bash
python main.py
```

3. **查看结果**
- 主要结果与图表保存在 `results/` 目录
- 完整实验报告见 `GNN_拓扑还原实验报告.md` 或 `.docx`

## 目录结构
```
gnn-0714/
├── src/                # 核心代码
├── main.py             # 主程序入口
├── config.json         # 配置文件
├── requirements.txt    # 依赖说明
├── results/            # 结果与图表
├── topologyzoo-main/   # 数据集
├── GNN_拓扑还原实验报告.md/.docx # 实验报告
├── 算法改进分析总结.md  # 算法对比与改进总结
├── 快速参考.md         # 关键用法速查
```

## 核心功能
- **一键实验**：自动完成数据加载、特征处理、模型训练与评估
- **多模型切换**：支持GCN、GAT、GraphSAGE等主流GNN
- **可视化输出**：自动生成训练曲线、模型对比等图表
- **灵活配置**：通过`config.json`自定义参数

## 主要实验成果
- **GraphSAGE模型**：AP=0.90，AUC=0.88（最佳）
- **GCN模型**：AP=0.79，AUC=0.81
- **GAT模型**：AP=0.50，AUC=0.50（需调优）
- **实验规模**：236个真实网络拓扑，结果具备统计意义

## 应用场景
- **运营商网络自动化运维**：自动发现和还原网络拓扑，辅助故障定位
- **网络规划与优化**：基于预测结果优化网络结构和资源配置
- **科研与教学**：真实数据驱动的GNN链路预测实验平台

## 常见问题
- **中文图表乱码**：请确保系统已安装“Microsoft YaHei”字体，或修改matplotlib字体设置
- **CUDA内存不足**：可切换为CPU训练或减少图规模
- **数据集加载失败**：请检查`topologyzoo-main/`目录及文件完整性

## 扩展与定制
- 支持自定义GNN模型与特征（详见`src/models.py`、`src/dataset.py`）
- 可扩展新数据集与评估指标
- 详细开发说明见`快速参考.md`

## 联系方式
如有问题或合作意向，请提交Issue或联系项目维护者。 