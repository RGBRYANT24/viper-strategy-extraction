# Delta-Uniform Self-Play for TicTacToe

## 快速开始

### 1. 测试安装
```bash
python test_delta_selfplay.py
```

### 2. 运行训练
```bash
# 基础版 (仅 Random 基准)
python train/train_delta_selfplay.py --total-timesteps 200000 --max-pool-size 20

# 完整版 (包含 MinMax 基准)
python train/train_delta_selfplay.py --total-timesteps 200000 --max-pool-size 20 --use-minmax
```

### 3. 查看详细文档
```bash
cat DELTA_SELFPLAY_GUIDE.md
```

## 实现的功能

✅ **基准策略** (`gym_env/policies/baseline_policies.py`)
- RandomPlayerPolicy: 随机策略
- MinMaxPlayerPolicy: 最优策略 (Alpha-Beta 剪枝)

✅ **Delta-Uniform 环境** (`gym_env/tictactoe_delta_selfplay.py`)
- 对手池管理 (基准池 + 学习池)
- 均匀采样机制
- 先后手训练 (视角翻转)

✅ **训练脚本** (`train/train_delta_selfplay.py`)
- 自动池管理 (固定大小 deque)
- 定期快照保存
- vs MinMax 评估

✅ **测试套件** (`test_delta_selfplay.py`)
- 基准策略测试
- 环境测试
- 注册测试
- 导入测试

## 解决的问题

| 问题 | 普通 Self-Play | Delta-Uniform Self-Play |
|------|----------------|------------------------|
| 陷入局部最优 | ❌ 易发生 | ✅ 不易发生 |
| 后手策略差 | ❌ 未训练后手 | ✅ 先后手均衡训练 |
| 对手单一 | ❌ 仅当前策略 | ✅ K+2 种对手 |
| 策略固定 | ❌ 缺乏多样性 | ✅ 历史池提供多样性 |

## 训练参数推荐

```bash
python train/train_delta_selfplay.py \
    --total-timesteps 200000 \    # 20万步 (约30-60分钟)
    --n-env 8 \                   # 8个并行环境
    --update-interval 10000 \     # 每1万步更新池
    --max-pool-size 20 \          # 保留20个历史快照
    --play-as-o-prob 0.5 \        # 先后手各50%
    --use-minmax                  # 包含MinMax基准
```

## 评估标准

**优秀策略** (vs MinMax 50局):
- 平局率: ≥ 90% (45-50局平局)
- 非法移动: 0

**良好策略**:
- 平局率: ≥ 60% (30-40局平局)
- 非法移动: 0

## 文件清单

```
新增文件:
├── gym_env/policies/
│   ├── __init__.py
│   └── baseline_policies.py
├── gym_env/tictactoe_delta_selfplay.py
├── train/train_delta_selfplay.py
├── test_delta_selfplay.py
├── DELTA_SELFPLAY_GUIDE.md
└── DELTA_SELFPLAY_README.md (本文件)

修改文件:
└── gym_env/__init__.py (注册新环境)
```

## 下一步

1. **运行测试**: `python test_delta_selfplay.py`
2. **开始训练**: 使用上述推荐参数
3. **评估模型**: 训练脚本会自动评估
4. **提取决策树**: 使用 VIPER 提取可解释策略

## 技术细节

详见 [DELTA_SELFPLAY_GUIDE.md](DELTA_SELFPLAY_GUIDE.md)
