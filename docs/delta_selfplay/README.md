# Delta-Uniform Self-Play 文档

## 文档索引

1. **[快速开始](DELTA_SELFPLAY_README.md)** - 安装、测试、运行训练
2. **[完整指南](DELTA_SELFPLAY_GUIDE.md)** - 详细实现、参数调优、常见问题
3. **[先后手训练机制](first_second_player.md)** - 视角翻转原理详解

## 核心文件

```
viper-verifiable-rl-impl/
├── gym_env/
│   ├── policies/
│   │   └── baseline_policies.py       # MinMax & Random 策略
│   └── tictactoe_delta_selfplay.py    # Delta-Uniform 环境
├── train/
│   └── train_delta_selfplay.py        # 训练脚本
└── test_delta_selfplay.py             # 测试套件
```

## 快速命令

```bash
# 测试
python test_delta_selfplay.py

# 训练 (推荐)
python train/train_delta_selfplay.py --total-timesteps 200000 --max-pool-size 20 --use-minmax

# 查看文档
cat docs/delta_selfplay/DELTA_SELFPLAY_README.md
```

## 关键特性

✅ 对手池机制 (K个历史快照 + 基准策略)
✅ 先后手均衡训练 (50%/50%)
✅ 视角翻转保证输入一致性
✅ 解决局部最优问题
