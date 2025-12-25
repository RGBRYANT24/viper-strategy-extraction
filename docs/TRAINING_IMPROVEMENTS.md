# PPO 训练改进说明

## 问题诊断

原始模型表现：
- ✅ vs Random: 98% 胜率
- ✅ vs MinMax: 100% 平局
- ❌ **战术知识: 33.3% 准确率** (严重问题)
- ❌ 不能识别立即获胜机会
- ❌ 不能识别关键防守局面

### 根本原因

1. **过拟合 MinMax**: 模型只学会了"不输"，没学会"主动进攻"
2. **探索不足**: 熵系数 0.01 太小，模型过早收敛到保守策略
3. **训练数据偏向**: MinMax 太强，导致模型只学防守
4. **奖励稀疏**: 只有终局奖励 (+1/-1/0)，无法区分"快速获胜"和"慢速获胜"

## 实施的改进

### 1. 增加探索性 (ent_coef)

**改动:**
```python
# 之前
ent_coef=0.01

# 之后
ent_coef=0.05  # 默认，可通过 --ent-coef 调整
```

**效果:**
- 鼓励模型尝试更多样的策略
- 避免过早收敛到局部最优
- 更容易发现"立即获胜"等战术机会

**建议值:**
- `0.05`: 标准探索 (推荐起点)
- `0.10`: 高探索 (如果模型过于保守)
- `0.02`: 低探索 (如果模型太随机)

### 2. 增加 Random 对手采样权重

**改动:**
```python
# 新增加权采样环境
class WeightedSelfPlayEnv(TicTacToeDeltaSelfPlayEnv):
    def _sample_opponent(self):
        # Random 对手权重: args.random_weight (默认 2.0)
        # MinMax 对手权重: 1.0
        # 学习池对手权重: 1.0
```

**效果:**
- Random 对手被采样的概率提高到 2倍
- 更多机会学习"如何快速击败弱对手"
- 平衡防守(vs MinMax)和进攻(vs Random)能力

**采样概率示例:**
```
假设对手池: [Random, MinMax, 学习池x10]
- 之前: Random=8.3%, MinMax=8.3%, 每个学习策略=8.3%
- 之后: Random=15.4%, MinMax=7.7%, 每个学习策略=7.7%
```

**建议值:**
- `1.0`: 均匀采样
- `2.0`: Random 翻倍 (推荐)
- `3.0`: Random 三倍 (如果进攻能力仍不足)
- `5.0`: 主要训练进攻

### 3. 始终包含 Random 对手

**改动:**
```python
# 之前: 只有指定 --use-minmax 才添加对手
baseline_policies = [RandomPlayerPolicy(obs_space, act_space)]
if args.use_minmax:
    baseline_policies.append(MinMaxPlayerPolicy(obs_space, act_space))

# 现在: Random 始终存在，MinMax 可选
```

**效果:**
- 确保模型始终能学习进攻策略
- 即使不使用 MinMax，也能训练出基本策略

## 使用方法

### 方案 1: 平衡训练 (推荐)

```bash
python train/train_delta_selfplay_ppo.py \
    --total-timesteps 300000 \
    --use-minmax \
    --ent-coef 0.05 \
    --random-weight 2.0 \
    --output log/oracle_TicTacToe_ppo_balanced.zip
```

适合: 需要同时学习进攻和防守

### 方案 2: 强化进攻能力

```bash
python train/train_delta_selfplay_ppo.py \
    --total-timesteps 300000 \
    --use-minmax \
    --ent-coef 0.10 \
    --random-weight 5.0 \
    --output log/oracle_TicTacToe_ppo_aggressive.zip
```

适合: 当前模型战术知识差，需要重点学习进攻

### 方案 3: 纯进攻训练

```bash
python train/train_delta_selfplay_ppo.py \
    --total-timesteps 200000 \
    --ent-coef 0.10 \
    --random-weight 1.0 \
    --output log/oracle_TicTacToe_ppo_offensive.zip
```

适合: 第一阶段训练，只学习进攻

### 方案 4: 分阶段训练 (最佳)

```bash
# 阶段 1: 学习进攻 (vs Random only)
python train/train_delta_selfplay_ppo.py \
    --total-timesteps 150000 \
    --ent-coef 0.10 \
    --random-weight 1.0 \
    --output log/stage1_offensive.zip

# 阶段 2: 学习防守 (加入 MinMax)
python train/train_delta_selfplay_ppo.py \
    --total-timesteps 150000 \
    --use-minmax \
    --ent-coef 0.05 \
    --random-weight 3.0 \
    --output log/stage2_balanced.zip
```

## 评估改进效果

```bash
# 运行完整评估
python train/evaluate_ppo_strategy.py \
    --model log/oracle_TicTacToe_ppo_balanced.zip \
    --num-games 100
```

**期望改进:**
- 战术知识准确率: 33% → 80%+
- vs Random 胜率: 98% → 98%+ (维持)
- vs MinMax 平局率: 100% → 90%+ (可接受下降)
- 立即获胜测试: 0/4 → 4/4 ✓
- 防守测试: 2/3 → 3/3 ✓

## 进一步改进方向

如果以上改进仍不足，可以考虑：

### 1. 奖励塑形 (Reward Shaping)

添加中间奖励：
- 制造威胁: +0.1
- 制造双威胁: +0.3
- 阻止对手: +0.2
- 快速获胜: +10 - 步数/10

### 2. 课程学习 (Curriculum Learning)

```
0-30% 训练: 只对抗 Random
30-60% 训练: Random + Weighted Random
60-100% 训练: Random + MinMax + 学习池
```

### 3. 增加网络容量

```bash
--net-arch 256,256  # 或 128,128,128
```

### 4. 延长训练时间

```bash
--total-timesteps 500000
```

## 参数调优指南

| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|----------|
| `--ent-coef` | 0.05 | 探索系数 | 模型保守→增加(0.10); 模型随机→减少(0.02) |
| `--random-weight` | 2.0 | Random权重 | 进攻弱→增加(3.0-5.0); 防守弱→减少(1.0) |
| `--total-timesteps` | 200000 | 总训练步数 | 性能不足→增加到300000-500000 |
| `--play-as-o-prob` | 0.5 | 后手概率 | 先手弱→0.3; 后手弱→0.7 |
| `--max-pool-size` | 20 | 策略池大小 | 多样性不足→增加到30-50 |

## 监控训练进度

训练过程中观察：
1. **对手池分布**: 确认 Random 被更频繁采样
2. **胜率变化**: vs Random 应该快速达到 >95%
3. **探索行为**: 前期应该看到较多不同的开局选择

训练后评估：
1. **战术知识**: 运行 `evaluate_ppo_strategy.py`
2. **对称性**: 检查旋转对称局面的决策
3. **实战测试**: 多次对战 MinMax 和 Random

## 常见问题

### Q: 为什么不直接用奖励塑形？
A: 奖励塑形需要仔细设计，容易引入偏差。先尝试简单改进（探索+采样），如果不够再加奖励塑形。

### Q: Random 权重设多少合适？
A: 取决于对手池大小：
- 只有 Random + MinMax: 2.0-3.0
- 有大量学习池(>10): 5.0-10.0

### Q: 训练多久合适？
A:
- 快速测试: 100k steps (~10分钟)
- 正常训练: 200k-300k steps (~30分钟)
- 完整训练: 500k steps (~1小时)

### Q: 如何判断是否收敛？
A: 运行评估脚本，检查：
- vs Random 胜率 >95%
- vs MinMax 平局率 >80%
- 战术知识准确率 >70%
- 对称性准确率 >80%

## 总结

三个核心改进：
1. ✅ **ent_coef: 0.01 → 0.05** (增加探索)
2. ✅ **random_weight: 1.0 → 2.0** (增加进攻训练)
3. ✅ **始终包含 Random 对手** (确保基础能力)

这些改进应该能显著提高模型的战术知识，特别是"立即获胜"和"主动进攻"能力，同时保持对 MinMax 的防守能力。
