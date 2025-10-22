# Delta-Uniform Self-Play 实现指南

## 概述

本实现针对 TicTacToe 提供了 **δ-Uniform Self-Play** 训练方法，用于解决普通自我对弈陷入局部最优的问题。

### 核心思想

1. **对手池机制**: 维护固定大小的历史策略池 (K个快照)
2. **基准策略**: 加入 MinMax 和 Random 策略提高鲁棒性
3. **均匀采样**: 每次 reset() 从池中均匀采样对手
4. **先后手训练**: 通过棋盘翻转保持网络输入一致性

### 与普通 Self-Play 的区别

| 特性 | 普通 Self-Play | Delta-Uniform Self-Play |
|------|----------------|------------------------|
| 对手来源 | 仅当前策略 | 历史池 + 基准池 |
| 对手多样性 | 低 (固定) | 高 (K+2 种) |
| 陷入局部最优 | 易发生 | 不易发生 |
| 先后手训练 | 通常只训练先手 | 先后手各 50% |

---

## 文件结构

```
viper-verifiable-rl-impl/
├── gym_env/
│   ├── policies/
│   │   ├── __init__.py
│   │   └── baseline_policies.py          # 基准策略 (MinMax, Random)
│   ├── tictactoe_delta_selfplay.py       # Delta-Uniform Self-Play 环境
│   └── __init__.py                        # 环境注册
├── train/
│   └── train_delta_selfplay.py           # 训练脚本
├── test_delta_selfplay.py                # 测试套件
└── DELTA_SELFPLAY_GUIDE.md               # 本文档
```

---

## 快速开始

### 1. 测试安装

运行测试套件验证所有组件正常工作:

```bash
python test_delta_selfplay.py
```

期望输出:
```
✓ 基准策略测试通过
✓ Delta-Uniform Self-Play 环境测试通过
✓ 环境注册测试通过
✓ 训练脚本导入测试通过
🎉 所有测试通过！可以开始训练了。
```

### 2. 开始训练

#### 基础训练 (仅 Random 基准)

```bash
python train/train_delta_selfplay.py \
    --total-timesteps 200000 \
    --max-pool-size 20 \
    --n-env 8
```

#### 高级训练 (包含 MinMax 基准)

```bash
python train/train_delta_selfplay.py \
    --total-timesteps 200000 \
    --max-pool-size 20 \
    --n-env 8 \
    --use-minmax
```

### 3. 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--total-timesteps` | 200000 | 总训练步数 |
| `--n-env` | 8 | 并行环境数 |
| `--update-interval` | 10000 | 策略池更新间隔 |
| `--max-pool-size` | 20 | 历史策略池容量 (K) |
| `--play-as-o-prob` | 0.5 | 后手训练概率 |
| `--use-minmax` | False | 是否包含 MinMax 基准 |
| `--output` | log/... | 模型保存路径 |

---

## 实现细节

### 1. 基准策略 (baseline_policies.py)

实现了两个基准策略,包装为 `stable-baselines3.BasePolicy` 接口:

#### RandomPlayerPolicy
- 从所有合法动作中随机选择
- 用于提供探索性对手

#### MinMaxPlayerPolicy
- 使用 Alpha-Beta 剪枝的 Minimax 算法
- 提供最优策略作为强对手
- 深度限制: 9 (完整搜索)

### 2. Delta-Uniform Self-Play 环境

#### 关键特性

**对手池管理**:
```python
baseline_pool = [RandomPolicy, MinMaxPolicy]  # 固定基准
learned_pool = deque(maxlen=K)  # 历史快照 (FIFO)
```

**采样机制**:
```python
def _sample_opponent():
    all_opponents = baseline_pool + list(learned_pool)
    return random.choice(all_opponents)  # 均匀采样
```

**先后手训练**:
```python
# 每局随机决定先后手
play_as_o = (random.random() < play_as_o_prob)

# 观察总是从自己视角 (自己=1, 对手=-1)
if play_as_o:
    obs = -board  # 翻转视角
else:
    obs = board
```

### 3. 训练流程

```
初始化:
├── 创建基准池: [Random, MinMax (可选)]
├── 创建学习池: deque(maxlen=K)
└── 创建 DQN 模型

训练循环 (每 10000 步):
├── 训练: model.learn(10000)
│   └── 环境 reset() 时自动采样对手
├── 更新池: learned_pool.append(copy.deepcopy(model.policy))
└── 统计: 打印池大小

评估:
└── 对战 MinMax 50局 (先手)
```

### 4. 视角一致性

神经网络必须看到一致的输入表示:

| 实际棋盘 | 我方标记 | 对方标记 | 网络输入 |
|----------|----------|----------|----------|
| 先手 (X) | 1 | -1 | board |
| 后手 (O) | -1 | 1 | -board |

通过翻转,网络总是看到 "自己=1, 对手=-1"。

---

## 训练建议

### 1. 池大小选择 (K)

| K 值 | 对手多样性 | 内存占用 | 推荐场景 |
|------|-----------|----------|----------|
| 10 | 低 | 低 | 快速实验 |
| 20 | 中 | 中 | **推荐** |
| 50 | 高 | 高 | 复杂游戏 |

### 2. 基准策略选择

**仅 Random**:
- 优点: 训练快,内存少
- 缺点: 可能学不到最优策略
- 适用: 初步实验

**Random + MinMax**:
- 优点: 能学到接近最优策略
- 缺点: MinMax 计算慢
- 适用: **生产环境**

### 3. 先后手比例

| play_as_o_prob | 说明 |
|----------------|------|
| 0.0 | 仅训练先手 (不推荐) |
| 0.5 | **均衡训练 (推荐)** |
| 1.0 | 仅训练后手 (不推荐) |

### 4. 超参数调优

**学习率**:
```python
learning_rate=1e-3  # 默认值
# 如果收敛慢: 1e-2
# 如果不稳定: 1e-4
```

**探索策略**:
```python
exploration_fraction=0.5  # 前 50% 时间探索
exploration_final_eps=0.05  # 最终 5% 探索率
```

---

## 评估指标

训练后会自动评估 50 局 vs MinMax (先手):

### 优秀策略 (目标)
```
胜: 0-2 (0-4%)
负: 0-2 (0-4%)
平: 46-50 (92-100%)  ← 关键指标
非法移动: 0
```

### 良好策略
```
平: 30-40 (60-80%)
非法移动: 0
```

### 需要改进
```
平: < 30 (< 60%)
或 非法移动 > 0
```

---

## 常见问题

### Q1: 训练后仍然陷入局部最优？

**可能原因**:
1. 池大小 K 太小 → 增加到 30-50
2. 未使用 MinMax 基准 → 添加 `--use-minmax`
3. 训练步数不足 → 增加到 300k-500k

### Q2: 对手池导致内存不足？

**解决方案**:
1. 减小 `--max-pool-size`
2. 增大 `--update-interval` (减少快照频率)
3. 使用更小的网络架构

### Q3: 训练速度太慢？

**优化建议**:
1. 不使用 MinMax 基准 (去掉 `--use-minmax`)
2. 减小并行环境数 `--n-env`
3. 减小池大小 `--max-pool-size`

### Q4: 如何对战自己训练的模型？

```python
from stable_baselines3 import DQN
import gymnasium as gym

# 加载模型
model = DQN.load("log/oracle_TicTacToe_delta_selfplay.zip")

# 创建环境
env = gym.make('TicTacToe-v0', opponent_type='random')

# 对战
obs, _ = env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    done = terminated or truncated

print(f"Result: {info}")
```

---

## 扩展到其他游戏

### 修改检查清单

1. **环境适配**:
   - [ ] 修改观察空间维度
   - [ ] 修改动作空间大小
   - [ ] 实现视角翻转逻辑

2. **基准策略**:
   - [ ] 实现随机策略
   - [ ] 实现规则策略 (如 MinMax)

3. **训练参数**:
   - [ ] 调整网络架构
   - [ ] 调整学习率
   - [ ] 调整池大小

---

## 参考文献

Delta-Uniform Self-Play 基于以下研究:

1. **PSRO (Policy-Space Response Oracles)**
   - Lanctot et al., 2017
   - 维护策略种群,计算近似纳什均衡

2. **Self-Play 改进方法**
   - Silver et al., AlphaGo/AlphaZero
   - 自我对弈 + MCTS

3. **Curriculum Learning in Multi-Agent RL**
   - Uniform sampling over historical opponents
   - 避免遗忘早期策略

---

## 贡献者

如有问题或建议,请提交 Issue 或 PR。

祝训练顺利! 🎉
