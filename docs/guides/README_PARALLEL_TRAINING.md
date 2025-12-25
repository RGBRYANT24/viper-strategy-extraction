# VIPER并行训练指南

本指南介绍如何在服务器上并行训练多个不同配置的VIPER决策树模型。

## 📋 目录

- [概述](#概述)
- [快速开始](#快速开始)
- [详细说明](#详细说明)
- [使用示例](#使用示例)
- [结果分析](#结果分析)
- [常见问题](#常见问题)

## 概述

该工具集支持在服务器上并行训练多个决策树配置，以测试不同的 `max_depth` 和 `max_leaves` 参数组合的性能。

**包含的工具：**

1. `train_parallel_viper.py` - 并行训练脚本
2. `analyze_viper_results.py` - 结果分析工具
3. `deploy_server.sh` - 服务器部署脚本（可选）

## 快速开始

### 1. 上传代码到服务器

```bash
# 在本地
scp -r /path/to/viper-verifiable-rl-impl user@server:/path/to/destination/
```

### 2. 在服务器上设置环境

```bash
# SSH到服务器
ssh user@server

# 进入项目目录
cd /path/to/viper-verifiable-rl-impl

# 安装依赖（如果还没安装）
pip install numpy torch gymnasium scikit-learn stable-baselines3 sb3-contrib joblib pandas
```

### 3. 运行并行训练

```bash
# 方法1: 使用部署脚本（推荐）
chmod +x deploy_server.sh
./deploy_server.sh --n-workers 8

# 方法2: 直接使用Python脚本
python train_parallel_viper.py \
    --depth-range 5 25 \
    --leaves-range 25 150 \
    --n-workers 8
```

### 4. 分析结果

```bash
python analyze_viper_results.py \
    --models-dir log/viper_mask_ppo_tictactoe \
    --n-episodes 1000
```

## 详细说明

### train_parallel_viper.py

并行训练多个配置的主脚本。

**关键参数：**

```bash
python train_parallel_viper.py \
    --depth-range MIN MAX        # 深度范围，如: 5 25
    --depth-step STEP            # 深度步长，默认: 5
    --leaves-range MIN MAX       # 叶子节点范围，如: 25 150
    --leaves-step STEP           # 叶子节点步长，默认: 25
    --n-iterations N             # 每个配置的VIPER迭代次数，默认: 10
    --samples-per-iter N         # 每轮采样数，默认: 50000
    --n-workers N                # 并行进程数，默认: CPU核心数-1
    --dry-run                    # 仅生成配置不训练
```

**示例：**

```bash
# 测试 depth=5,10,15,20,25 和 leaves=25,50,75,100,125,150 的所有组合
python train_parallel_viper.py \
    --depth-range 5 25 --depth-step 5 \
    --leaves-range 25 150 --leaves-step 25 \
    --n-iterations 10 \
    --samples-per-iter 50000 \
    --n-workers 4

# 这会训练 5 x 6 = 30 个配置
```

**输出：**

- 模型文件：`log/viper_mask_ppo_tictactoe/viper_mask_ppo_tree_*.joblib`
- 训练日志：`log/viper_mask_ppo_tictactoe/viper_mask_ppo_tree_*_training_log.txt`
- 配置文件：`log/viper_parallel_training/configs_*.json`
- 结果摘要：`log/viper_parallel_training/results_*.json`

### analyze_viper_results.py

分析和对比所有训练好的模型。

**关键参数：**

```bash
python analyze_viper_results.py \
    --models-dir DIR             # 模型目录，默认: log/viper_mask_ppo_tictactoe
    --pattern PATTERN            # 模型文件名模式，默认: viper_mask_ppo_tree_*.joblib
    --n-episodes N               # 评估局数，默认: 1000
    --opponents TYPE [TYPE ...]  # 对手类型，默认: random minmax
    --output-dir DIR             # 输出目录，默认: log/viper_parallel_training
```

**示例：**

```bash
# 评估所有模型，每个对手1000局
python analyze_viper_results.py \
    --models-dir log/viper_mask_ppo_tictactoe \
    --n-episodes 1000 \
    --opponents random minmax

# 只分析已有结果，不重新评估
python analyze_viper_results.py --no-evaluate
```

**输出：**

- 对比报告：`log/viper_parallel_training/report_*.txt`
- CSV数据：`log/viper_parallel_training/results_*.csv`

### deploy_server.sh

一键部署和训练脚本（可选）。

**使用：**

```bash
chmod +x deploy_server.sh

# 使用默认配置
./deploy_server.sh

# 自定义配置
./deploy_server.sh \
    --depth-min 5 --depth-max 25 --depth-step 5 \
    --leaves-min 25 --leaves-max 150 --leaves-step 25 \
    --n-iterations 10 \
    --samples-per-iter 50000 \
    --n-workers 8

# 测试配置（不实际训练）
./deploy_server.sh --dry-run
```

## 使用示例

### 示例1：小规模测试

```bash
# 快速测试少量配置
python train_parallel_viper.py \
    --depth-range 5 15 --depth-step 5 \
    --leaves-range 25 75 --leaves-step 25 \
    --n-iterations 5 \
    --samples-per-iter 10000 \
    --n-workers 2

# 配置数：3 x 3 = 9
```

### 示例2：完整实验

```bash
# 训练完整的参数网格
python train_parallel_viper.py \
    --depth-range 5 25 --depth-step 5 \
    --leaves-range 25 150 --leaves-step 25 \
    --n-iterations 10 \
    --samples-per-iter 50000 \
    --n-workers 8

# 配置数：5 x 6 = 30
```

### 示例3：使用nohup后台运行

```bash
# 在服务器上后台运行，防止SSH断开
nohup python train_parallel_viper.py \
    --depth-range 5 25 \
    --leaves-range 25 150 \
    --n-workers 8 \
    > training.log 2>&1 &

# 查看进度
tail -f training.log

# 查看后台任务
jobs

# 检查进程
ps aux | grep train_parallel_viper
```

### 示例4：使用screen会话

```bash
# 创建新的screen会话
screen -S viper_training

# 在screen中运行训练
python train_parallel_viper.py \
    --depth-range 5 25 \
    --leaves-range 25 150 \
    --n-workers 8

# 离开screen: Ctrl+A, D
# 重新连接: screen -r viper_training
# 终止screen: Ctrl+A, K
```

## 结果分析

### 查看训练日志

每个模型都有独立的训练日志：

```bash
# 查看特定模型的训练日志
cat log/viper_mask_ppo_tictactoe/viper_mask_ppo_tree_20251107_003010_iter10_samples50000_depth15_leaves100_training_log.txt
```

### 生成对比报告

```bash
# 生成完整的对比报告
python analyze_viper_results.py \
    --models-dir log/viper_mask_ppo_tictactoe \
    --n-episodes 1000

# 报告包含：
# - 最佳模型（vs Random, vs MinMax）
# - 所有模型性能表格
# - 统计分析
```

### 报告示例

```
==========================================================================================
VIPER模型性能对比报告
==========================================================================================
生成时间: 2025-11-07 14:30:00
总模型数: 30

配置分布:
  深度范围: 5 - 25
  叶子节点范围: 25 - 150

------------------------------------------------------------------------------------------
对战 RANDOM 最佳模型 (按胜率)
------------------------------------------------------------------------------------------
模型: viper_mask_ppo_tree_20251107_003010_iter10_samples50000_depth15_leaves100.joblib
配置: depth=15, leaves=100
胜率: 98.50%
平局率: 1.20%
负率: 0.30%
平均奖励: 0.9820

------------------------------------------------------------------------------------------
对战 MINMAX 最佳模型 (按平局率)
------------------------------------------------------------------------------------------
模型: viper_mask_ppo_tree_20251107_003045_iter10_samples50000_depth15_leaves150.joblib
配置: depth=15, leaves=150
胜率: 0.00%
平局率: 92.50%
负率: 7.50%
平均奖励: -0.0750

------------------------------------------------------------------------------------------
所有模型性能汇总
------------------------------------------------------------------------------------------
Depth | Leaves | Random (W/D) | MinMax (W/D)
---------------------------------------------
    5 |     25 | 95.2%/ 3.5% |  0.0%/85.0%
    5 |     50 | 96.8%/ 2.8% |  0.0%/87.5%
   10 |     25 | 97.5%/ 2.0% |  0.0%/88.0%
   10 |     50 | 98.2%/ 1.5% |  0.0%/90.5%
   15 |    100 | 98.5%/ 1.2% |  0.0%/91.0%
   15 |    150 | 98.3%/ 1.4% |  0.0%/92.5%
   ...
```

### 使用CSV进行自定义分析

```python
import pandas as pd
import matplotlib.pyplot as plt

# 加载结果
df = pd.read_csv('log/viper_parallel_training/results_20251107_143000.csv')

# 绘制胜率热图
pivot = df.pivot(index='depth', columns='leaves', values='random_win_rate')
plt.figure(figsize=(10, 6))
plt.imshow(pivot, cmap='YlGnBu', aspect='auto')
plt.colorbar(label='Win Rate')
plt.xlabel('Leaves')
plt.ylabel('Depth')
plt.title('Win Rate vs Random by Tree Configuration')
plt.savefig('win_rate_heatmap.png')
```

## 常见问题

### Q1: 如何确定合适的并行进程数？

```bash
# 查看CPU核心数
nproc

# 建议使用 CPU核心数 - 1 或 CPU核心数 / 2
# 例如32核服务器：
python train_parallel_viper.py --n-workers 16
```

### Q2: 训练时间估算

单个配置训练时间 ≈ n_iterations × samples_per_iter / 采样速度

- 采样速度约为 1000-5000 samples/second（取决于硬件）
- 例如：10 iterations × 50000 samples = 500000 samples
- 时间：500000 / 3000 ≈ 167秒 ≈ 3分钟/配置

并行训练30个配置，使用8进程：
- 总时间 ≈ (30 / 8) × 3分钟 ≈ 12分钟

### Q3: 内存不足怎么办？

```bash
# 减少并行进程数
python train_parallel_viper.py --n-workers 2

# 或减少每轮采样数
python train_parallel_viper.py --samples-per-iter 20000

# 监控内存使用
htop  # 或 top
```

### Q4: 如何恢复中断的训练？

训练脚本会为每个配置独立训练，如果部分配置失败或中断：

1. 查看结果摘要找出失败的配置
2. 手动训练失败的配置：

```bash
python train/viper_mask_ppo.py \
    --mode train \
    --max_depth 15 \
    --max_leaves 100 \
    --n_iterations 10 \
    --samples_per_iter 50000
```

### Q5: 如何只评估已有模型？

```bash
# 跳过训练，只分析已有模型
python analyze_viper_results.py \
    --models-dir log/viper_mask_ppo_tictactoe \
    --no-evaluate
```

## 文件结构

```
viper-verifiable-rl-impl/
├── train_parallel_viper.py          # 并行训练脚本
├── analyze_viper_results.py         # 结果分析脚本
├── deploy_server.sh                 # 部署脚本
├── train/
│   └── viper_mask_ppo.py           # VIPER训练核心代码
├── log/
│   ├── viper_mask_ppo_tictactoe/   # 模型和训练日志
│   │   ├── viper_mask_ppo_tree_*.joblib
│   │   └── viper_mask_ppo_tree_*_training_log.txt
│   └── viper_parallel_training/    # 并行训练摘要
│       ├── configs_*.json
│       ├── results_*.json
│       ├── results_*.csv
│       └── report_*.txt
└── README_PARALLEL_TRAINING.md     # 本文档
```

## 性能优化建议

1. **合理设置进程数**
   - 不要超过CPU核心数
   - 留一些核心给系统

2. **调整采样数**
   - 初期测试可以用较少的采样数（如10000）
   - 正式实验使用50000或更多

3. **使用GPU加速**
   - 如果有GPU，确保PyTorch能识别
   - 设置环境变量：`export CUDA_VISIBLE_DEVICES=0,1`

4. **监控资源使用**
   ```bash
   # CPU和内存
   htop

   # GPU使用
   nvidia-smi -l 1

   # 磁盘I/O
   iostat -x 1
   ```

## 联系和支持

如有问题，请查看：
- [test_trees.py](test_trees.py) - 原始评估脚本
- [train/viper_mask_ppo.py](train/viper_mask_ppo.py) - VIPER核心实现
