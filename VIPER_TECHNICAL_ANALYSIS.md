# VIPER MaskablePPO 实现技术分析文档

本文档详细分析 `viper_maskable_ppo.py` 的实现思路，与 VIPER 原论文的对比，以及技术细节。

---

## 📚 目录

1. [VIPER 原论文算法回顾](#1-viper-原论文算法回顾)
2. [本实现与原论文的对比](#2-本实现与原论文的对比)
3. [训练对手配置说明](#3-训练对手配置说明)
4. [核心代码逐段解析](#4-核心代码逐段解析)
5. [关键技术细节](#5-关键技术细节)
6. [与项目中 DQN-VIPER 的对比](#6-与项目中-dqn-viper-的对比)
7. [常见问题与解决方案](#7-常见问题与解决方案)

---

## 1. VIPER 原论文算法回顾

### 1.1 论文基本信息

**论文**: Verifiable Reinforcement Learning via Policy Extraction
**作者**: Osbert Bastani et al. (2018)
**链接**: https://arxiv.org/abs/1805.08328

### 1.2 核心算法（伪代码）

```python
Algorithm: VIPER (Verifiable Policy Extraction via Reinforcement learning)

Input:
  - π_oracle: Pre-trained DQN/PPO oracle policy
  - M: MDP environment
  - N: Number of iterations
  - n: Number of samples per iteration

Output:
  - π_tree: Interpretable decision tree policy

Procedure:
  D ← ∅  # Training dataset
  π_current ← π_oracle  # Current policy (initially oracle)

  for i = 1 to N do:
    # Step 1: Sample trajectories using mixture of oracle and current tree
    β ← 1 if i == 1 else 0  # DAgger aggregation parameter
    τ ← Sample_Trajectories(π_current, π_oracle, β, n)

    # Step 2: Compute criticality weights for each state
    for (s, a, r) in τ do:
      l(s) ← Criticality(π_oracle, s)  # Q_max - Q_min or log_prob_max - log_prob_min
      D ← D ∪ {(s, a, l(s))}

    # Step 3: Train decision tree with weighted samples
    π_tree_i ← Train_DecisionTree(D, sample_weight=l(s))

    # Step 4: Evaluate and track best tree
    R_i ← Evaluate(π_tree_i, M)
    π_current ← π_tree_i

  return Best(π_tree_1, ..., π_tree_N)
```

### 1.3 核心思想

1. **Oracle Teacher**: 使用高性能神经网络（DQN/PPO）作为教师
2. **Imitation Learning**: 决策树模仿 Oracle 的行为
3. **Criticality Weighting**: 重要状态（决策影响大）给予更高权重
4. **Iterative Refinement**: 多轮迭代，选择最佳树
5. **DAgger-style Aggregation**: 混合使用 Oracle 和当前策略采样

---

## 2. 本实现与原论文的对比

### 2.1 核心一致性 ✅

| 维度 | 原论文 | 本实现 | 一致性 |
|------|--------|--------|--------|
| **算法框架** | DAgger + Weighted Imitation | DAgger + Weighted Imitation | ✅ 完全一致 |
| **Criticality Loss** | Q_max - Q_min (DQN)<br>log_prob_max - log_prob_min (PPO) | 同左 | ✅ 完全一致 |
| **Beta 调度** | β=1 (iter 0), β=0 (iter 1+) | 同左 | ✅ 完全一致 |
| **树模型** | DecisionTreeClassifier | 同左 | ✅ 完全一致 |
| **样本权重** | sample_weight=criticality | 同左 | ✅ 完全一致 |
| **迭代选择** | 选择 N 棵树中最佳 | 同左 | ✅ 完全一致 |

### 2.2 实现增强 🔧

| 增强点 | 原论文 | 本实现 | 说明 |
|--------|--------|--------|------|
| **Oracle 类型** | DQN, 传统 PPO | **MaskablePPO** | ✅ 支持 action masking |
| **非法动作** | 未明确处理 | **Mask 处理** | ✅ 100% 避免非法动作 |
| **输出方式** | 单个动作 | **概率分布 + Masking** | ✅ 更符合 TicTacToe |
| **环境兼容性** | Gym | **Gymnasium + ActionMasker** | ✅ 适配新版 API |

### 2.3 关键差异分析

#### 差异 1: Criticality Loss 计算中考虑 Action Masking

**原论文**:
```python
# DQN
Q_values = oracle.q_net(obs)
criticality = Q_values.max() - Q_values.min()  # 所有动作

# PPO
log_probs = [oracle.policy.log_prob(obs, a) for a in all_actions]
criticality = log_probs.max() - log_probs.min()  # 所有动作
```

**本实现**:
```python
# MaskablePPO (仅考虑合法动作)
log_probs = oracle.policy.get_distribution(obs).logits
legal_actions = where(obs == 0)  # 获取合法动作
legal_log_probs = log_probs[legal_actions]
criticality = legal_log_probs.max() - legal_log_probs.min()  # ⭐ 仅合法动作
```

**为什么重要**:
- TicTacToe 非法动作的 Q 值/log_prob 无意义
- 仅考虑合法动作更准确反映状态重要性

#### 差异 2: 决策树输出方式

**原论文**:
```python
# 直接输出单个动作
action = tree.predict(obs)
```

**本实现**:
```python
# 输出概率分布，然后应用 masking
probs = tree.predict_proba(obs)  # [p0, p1, ..., p8]
legal_actions = where(obs == 0)
masked_probs = probs.copy()
masked_probs[illegal_actions] = -inf
action = argmax(masked_probs)  # ⭐ 保证合法
```

**为什么重要**:
- 保证推理时 100% 避免非法动作
- 不需要在训练时特殊处理非法动作标签

### 2.4 算法核心思想一致性 ✅

| 核心思想 | 是否保持 |
|----------|---------|
| 使用高性能 Oracle 作为教师 | ✅ 是 (MaskablePPO) |
| 基于 Criticality 加权样本 | ✅ 是 (考虑 masking) |
| DAgger-style 数据聚合 | ✅ 是 (β=1 then 0) |
| 迭代训练多棵树并选最佳 | ✅ 是 |
| 提取可解释决策树 | ✅ 是 (单棵分类树) |

**结论**: 本实现在保持 VIPER 核心思想的基础上，针对 TicTacToe 和 MaskablePPO 做了必要的适配，**没有偏离原论文的核心算法**。

---

## 3. 训练对手配置说明

### 3.1 当前实现的对手设置

**代码位置**: `load_maskable_ppo_oracle()` 函数

```python
def load_maskable_ppo_oracle(oracle_path, env_name, opponent_type='minmax'):
    """
    Args:
        opponent_type: 对手类型 ('random', 'minmax')
    """
    env = gym.make(env_name, opponent_type=opponent_type)
    # ...
```

**默认配置**: `opponent_type='minmax'`

**问题**: 确实，VIPER 训练时**只使用单一对手**（默认 MinMax）。

### 3.2 为什么只用一个对手？

#### 原因 1: Oracle 训练 vs VIPER 训练的区别

| 阶段 | 训练方式 | 对手配置 | 目的 |
|------|----------|----------|------|
| **Oracle 训练** (PPO) | Self-play + 多对手池 | Random + MinMax + 历史策略 | 学习鲁棒策略 |
| **VIPER 训练** (Tree) | 模仿 Oracle | **单一固定对手** | 提取已学知识 |

**关键点**: VIPER 不是从零学习，而是**提取 Oracle 已经学到的知识**。Oracle 已经在多样化对手上训练过，所以 VIPER 只需要在一个固定对手上提取规则。

#### 原因 2: VIPER 论文的设计哲学

VIPER 的核心是**模仿学习 (Imitation Learning)**，而不是强化学习：
- Oracle 已经知道如何应对各种对手
- VIPER 只需要学习"Oracle 会怎么做"
- 对手只是提供状态分布的采样环境

#### 原因 3: 计算效率

使用单一对手：
- ✅ 更快的采样速度
- ✅ 更稳定的状态分布
- ✅ 更容易评估收敛性

### 3.3 应该改成多对手吗？

**理论上可以，但不是必需**。

#### 方案 A: 保持单对手（推荐）⭐

```python
# 使用 MinMax（最强对手）
python train/viper_maskable_ppo.py \
    --oracle-path log/oracle_TicTacToe_ppo_aggressive.zip \
    --opponent-type minmax \
    --total-timesteps 50000
```

**优势**:
- ✅ 简单稳定
- ✅ 采样高质量状态
- ✅ 符合 VIPER 原论文设计

#### 方案 B: 多对手采样（可选增强）

如果想要更鲁棒的树，可以修改代码支持多对手：

```python
# 伪代码（需要修改实现）
opponents = ['random', 'minmax']
for iteration in range(n_iter):
    opponent = random.choice(opponents)  # 每轮随机对手
    env = gym.make(env_name, opponent_type=opponent)
    trajectory = sample_trajectory(...)
```

**但这不是 VIPER 原论文的做法**，而是额外增强。

### 3.4 推荐配置

| 场景 | 推荐对手 | 原因 |
|------|----------|------|
| **标准训练** | `minmax` | 最强对手，提取最优策略 |
| **快速测试** | `random` | 更简单，采样更快 |
| **高鲁棒性** | 修改代码支持多对手 | 实验性，非必需 |

**结论**: 当前实现使用单一对手是**符合 VIPER 论文设计**的，不是 bug 或遗漏。

---

## 4. 核心代码逐段解析

### 4.1 算法流程概览

```
┌─────────────────────────────────────────────────────────────┐
│                    VIPER 训练流程                            │
└─────────────────────────────────────────────────────────────┘

1. 加载 Oracle (MaskablePPO)
   ↓
2. 初始化空数据集 D = []
   ↓
3. FOR iteration = 1 to N:
   │
   ├─→ 3.1 确定采样策略
   │        β = 1 (iter 0) or 0 (iter 1+)
   │
   ├─→ 3.2 采样轨迹
   │        - 使用 β-weighted 混合策略
   │        - 获取 (state, oracle_action, criticality)
   │
   ├─→ 3.3 聚合数据
   │        D = D ∪ new_trajectory
   │
   ├─→ 3.4 训练决策树
   │        tree = DecisionTreeClassifier()
   │        tree.fit(X, y, sample_weight=criticality)
   │
   ├─→ 3.5 评估树性能
   │        reward = evaluate(tree)
   │
   └─→ 3.6 更新当前策略
            policy = tree
   │
4. 返回最佳树 (max reward)
```

### 4.2 关键函数详解

#### 函数 1: `mask_fn(env)` - 获取动作掩码

```python
def mask_fn(env):
    """
    返回 action mask for TicTacToe

    处理多层包装器: ActionMasker -> Monitor -> TicTacToeEnv
    """
    current_env = env
    max_depth = 10
    depth = 0

    while depth < max_depth:
        if hasattr(current_env, 'board'):
            # 找到 TicTacToe 环境
            board = current_env.board
            mask = (board == 0).astype(np.int8)
            return mask
        elif hasattr(current_env, 'env'):
            # 继续解包
            current_env = current_env.env
            depth += 1
        else:
            break

    raise AttributeError("Cannot find 'board' attribute")
```

**作用**:
- 从多层包装的环境中提取棋盘状态
- 返回合法动作掩码 (1=合法, 0=非法)

**为什么需要递归解包**:
```
ActionMasker(
  Monitor(
    TicTacToeEnv()  ← 真正的环境，包含 board 属性
  )
)
```

#### 函数 2: `ProbabilityMaskedTreeWrapper` - 决策树包装器

```python
class ProbabilityMaskedTreeWrapper:
    def __init__(self, tree_model):
        self.tree = tree_model  # sklearn DecisionTreeClassifier
        self.n_actions = 9

    def predict(self, observation, ...):
        """
        核心推理逻辑：概率分布 + Masking
        """
        # 1. 获取所有动作的概率分布
        action_probs = self.tree.predict_proba(observation)  # shape: (batch, 9)

        # 2. 对每个样本应用 masking
        for i in range(observation.shape[0]):
            obs = observation[i]
            probs = action_probs[i]

            # 3. 获取合法动作（棋盘上的空位）
            legal_actions = np.where(obs == 0)[0]

            # 4. 创建掩码概率（非法动作 = -inf）
            masked_probs = np.full(9, -np.inf)
            masked_probs[legal_actions] = probs[legal_actions]

            # 5. 选择概率最高的合法动作
            action = np.argmax(masked_probs)
            actions.append(action)

        return actions
```

**关键优势**:
- ✅ 保证 100% 合法动作
- ✅ 不需要训练时处理非法动作标签
- ✅ 保持决策树的可解释性

**与原论文对比**:
- 原论文: 直接 `tree.predict(obs)`
- 本实现: `tree.predict_proba(obs)` + masking
- **原因**: TicTacToe 需要强制避免非法动作

#### 函数 3: `get_criticality_loss_maskable_ppo()` - 计算重要性权重

```python
def get_criticality_loss_maskable_ppo(oracle, observations):
    """
    计算状态的 criticality（重要性）

    公式: criticality(s) = max_a∈Legal Q(s,a) - min_a∈Legal Q(s,a)

    其中 Q(s,a) ≈ log π(a|s) (max entropy formulation)
    """
    with torch.no_grad():
        # 1. 获取 log probabilities (近似 Q 值)
        obs_tensor = torch.as_tensor(observations).to(oracle.device)
        distribution = oracle.policy.get_distribution(obs_tensor)
        log_probs = distribution.distribution.logits.cpu().numpy()  # (batch, 9)

        # 2. 对每个状态计算 masked criticality
        losses = []
        for i in range(observations.shape[0]):
            obs = observations[i]
            log_prob = log_probs[i]

            # 3. 只考虑合法动作
            legal_actions = np.where(obs == 0)[0]

            if len(legal_actions) == 0:
                losses.append(0.0)  # 无合法动作（不应该发生）
            else:
                # 4. 计算合法动作的 Q 值范围
                legal_log_probs = log_prob[legal_actions]
                criticality = legal_log_probs.max() - legal_log_probs.min()
                losses.append(criticality)

        return np.array(losses)
```

**核心思想**:
1. **Q ≈ log π**: PPO 的策略概率近似 Q 值（max entropy RL）
2. **Criticality = Range**: 最佳动作和最差动作的差距
3. **高 criticality** → 决策很重要（权重高）
4. **低 criticality** → 随便选都行（权重低）

**示例**:
```python
# 状态 1: 即将获胜
legal_actions = [2, 4]  # 两个空位
log_probs = {2: -0.1, 4: -3.0}  # 位置2几乎确定赢
criticality = -0.1 - (-3.0) = 2.9  # 高权重 ⭐

# 状态 2: 开局第一步
legal_actions = [0,1,2,3,4,5,6,7,8]
log_probs = [-1.2, -1.3, -1.1, ...]  # 差不多
criticality = -1.1 - (-1.3) = 0.2  # 低权重
```

#### 函数 4: `sample_trajectory_maskable_ppo()` - 采样轨迹

```python
def sample_trajectory_maskable_ppo(oracle, env, policy, beta, n_steps, verbose=0):
    """
    使用 beta-weighted 混合策略采样轨迹

    Args:
        oracle: MaskablePPO 教师模型
        env: 环境（带 ActionMasker）
        policy: 当前决策树（或 None）
        beta: Oracle 采样概率 (1=全oracle, 0=全tree)
        n_steps: 采样步数

    Returns:
        trajectory: [(obs, oracle_action, criticality_weight), ...]
    """
    trajectory = []
    obs, _ = env.reset()

    while len(trajectory) < n_steps:
        # 1. 选择策略：以 beta 概率使用 oracle
        use_oracle = (policy is None) or (np.random.random() < beta)

        # 2. 获取 action mask
        action_mask = mask_fn(env)

        # 3. 执行动作
        if use_oracle:
            action, _ = oracle.predict(obs, action_masks=action_mask)
        else:
            action, _ = policy.predict(obs)  # tree 内置 masking

        # 4. 获取 oracle 的标签（用于训练）
        oracle_action, _ = oracle.predict(obs, action_masks=action_mask)

        # 5. 环境交互
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 6. 计算 criticality 权重
        obs_batch = obs.reshape(1, -1)
        state_loss = get_criticality_loss_maskable_ppo(oracle, obs_batch)[0]

        # 7. 添加样本 (状态, oracle动作, 权重)
        trajectory.append((obs.copy(), oracle_action, state_loss))

        obs = next_obs
        if done:
            obs, _ = env.reset()

    return trajectory
```

**Beta 调度说明**:
```python
# Iteration 0: beta = 1.0
#   → 100% 使用 oracle 采样
#   → 目的: 收集高质量初始数据

# Iteration 1+: beta = 0.0
#   → 100% 使用当前 tree 采样
#   → 目的: DAgger-style 数据聚合
#   → 修正 tree 的错误分布
```

**为什么需要 oracle_action 作为标签**:
- 即使用 tree 采样，也要学习 oracle 的决策
- 这样 tree 逐渐修正自己的错误

#### 函数 5: `train_viper_maskable_ppo()` - 主训练循环

```python
def train_viper_maskable_ppo(args):
    """
    VIPER 主训练流程
    """
    # 1. 加载 Oracle
    env, oracle = load_maskable_ppo_oracle(
        args.oracle_path, args.env_name, args.opponent_type
    )

    # 2. 初始化
    dataset = []
    policy = None
    policies = []
    rewards = []

    n_steps_per_iter = args.total_timesteps // args.n_iter

    # 3. VIPER 迭代
    for iteration in range(args.n_iter):
        # 3.1 确定 beta
        beta = 1.0 if iteration == 0 else 0.0

        # 3.2 采样轨迹
        trajectory = sample_trajectory_maskable_ppo(
            oracle, env, policy, beta, n_steps_per_iter
        )
        dataset += trajectory

        # 3.3 准备训练数据
        X = np.array([traj[0] for traj in dataset])      # 状态
        y = np.array([traj[1] for traj in dataset])      # oracle 动作
        weights = np.array([traj[2] for traj in dataset]) # criticality

        # 3.4 训练决策树
        clf = DecisionTreeClassifier(
            ccp_alpha=0.0001,      # 剪枝参数
            criterion="entropy",    # 信息增益
            max_depth=args.max_depth,
            max_leaf_nodes=args.max_leaves,
            random_state=42
        )
        clf.fit(X, y, sample_weight=weights)  # ⭐ 加权训练

        # 3.5 包装和评估
        wrapped_policy = ProbabilityMaskedTreeWrapper(clf)
        policies.append(clf)
        policy = wrapped_policy

        eval_env = gym.make(args.env_name, opponent_type=args.opponent_type)
        mean_reward, std_reward = evaluate_policy(wrapped_policy, eval_env, n_eval_episodes=100)
        rewards.append(mean_reward)

    # 4. 选择最佳树
    best_idx = np.argmax(rewards)
    best_policy = policies[best_idx]

    return ProbabilityMaskedTreeWrapper(best_policy)
```

**关键点**:
1. **数据累积**: `dataset += trajectory` (不是 `dataset = trajectory`)
2. **加权训练**: `sample_weight=weights`
3. **迭代选择**: 返回 N 棵树中性能最好的

---

## 5. 关键技术细节

### 5.1 为什么 Beta = 1 (iter 0) then 0 (iter 1+)?

这是 **DAgger (Dataset Aggregation)** 算法的核心思想。

#### 问题背景

**普通模仿学习**的问题:
```python
# 只用 oracle 采样
for iteration in range(N):
    trajectory = sample(oracle)  # 始终用 oracle
    tree.fit(trajectory)
```

**问题**: Tree 的错误会累积：
```
Step 1: Tree 预测错 → 进入 oracle 没见过的状态
Step 2: Tree 不知道怎么办 → 继续预测错
Step 3: 状态越来越偏离 oracle 的分布 → 性能崩溃
```

这叫做 **Covariate Shift（协变量偏移）**。

#### DAgger 的解决方案

```python
# Iteration 0: 用 oracle 采样
beta = 1.0  # 100% oracle
trajectory = sample(oracle)  # 高质量初始数据

# Iteration 1+: 用 tree 采样
beta = 0.0  # 100% tree
trajectory = sample(tree)  # Tree 自己的状态分布
# 但标签仍然是 oracle 的动作！

# 这样 tree 在自己会遇到的状态上学习 oracle 的行为
```

**效果**:
- ✅ Tree 学习如何从自己的错误中恢复
- ✅ 修正状态分布偏移
- ✅ 更鲁棒的策略

### 5.2 Criticality Loss 的数学原理

#### Max-Entropy RL 的 Q 函数近似

在 Max-Entropy RL 框架下（PPO 使用此框架）：

```
Q(s, a) ≈ log π(a|s) + const
```

**原因**: 最优策略满足
```
π*(a|s) ∝ exp(Q(s,a) / α)
⇒ log π*(a|s) = Q(s,a) / α - log Z(s)
⇒ Q(s,a) ≈ α · log π(a|s) + const
```

其中 α 是温度参数（通常为1），Z(s) 是归一化常数。

#### Criticality 定义

```
Criticality(s) = max_a Q(s,a) - min_a Q(s,a)
               ≈ max_a log π(a|s) - min_a log π(a|s)
```

**直观理解**:
- **高 criticality**: 选对动作很重要（差距大）
- **低 criticality**: 选什么都差不多（差距小）

#### 为什么只考虑合法动作?

```python
# 错误方式（包含非法动作）
all_log_probs = [-0.5, -1.2, -5.0, -5.0, -5.0, ...]  # 后面是非法动作
criticality = -0.5 - (-5.0) = 4.5  # 被非法动作污染！

# 正确方式（仅合法动作）
legal_log_probs = [-0.5, -1.2]  # 只有合法动作
criticality = -0.5 - (-1.2) = 0.7  # 真实差距
```

### 5.3 决策树训练的权重作用

```python
clf.fit(X, y, sample_weight=weights)
```

**Weighted Loss**:
```
Loss = Σ weight[i] × CrossEntropy(y_pred[i], y_true[i])
     = Σ criticality[i] × CE(...)
```

**效果**:
- 重要状态的错误被惩罚得更重
- 不重要状态的错误影响较小
- Tree 优先学好关键决策点

**示例**:
```
# 数据集
State 1 (开局):   weight = 0.2  (不重要)
State 2 (中局):   weight = 1.5  (一般)
State 3 (决胜):   weight = 8.0  (非常重要！)

# 如果 State 3 预测错，Loss 增加 8.0 × CE
# 如果 State 1 预测错，Loss 只增加 0.2 × CE
# → Tree 会优先学好 State 3
```

### 5.4 为什么选择单棵分类树而不是回归树?

项目中有两种方案：
1. **单棵分类树** (本实现) ← 推荐
2. **多棵回归树** (archive/regression_tree_approach/)

#### 对比

| 维度 | 分类树 | 回归树 (9棵) |
|------|--------|-------------|
| **输出** | 动作类别 (0-8) | 每个动作的 Q 值 |
| **可解释性** | ✅ 完整 IF-THEN 规则 | ✗ 9棵树，难以理解 |
| **模型大小** | ✅ 1棵树 | ✗ 9棵树 |
| **非法动作** | ✅ Masking 处理 | △ 需要额外逻辑 |
| **精度** | △ 可能略低 | ✅ 可能略高 |

**为什么分类树更好**:
1. VIPER 的核心目标是**可解释性**，不是性能
2. 单棵树可以完整提取决策逻辑
3. 配合概率掩码，不牺牲正确性

---

## 6. 与项目中 DQN-VIPER 的对比

### 6.1 代码对比

| 组件 | DQN-VIPER (`train/viper.py`) | MaskablePPO-VIPER (`train/viper_maskable_ppo.py`) |
|------|------------------------------|--------------------------------------------------|
| **Oracle** | `stable_baselines3.DQN` | `sb3_contrib.MaskablePPO` |
| **环境** | 直接 `gym.make()` | `gym.make()` + `ActionMasker` |
| **Criticality** | `Q_max - Q_min` (所有动作) | `log_prob_max - log_prob_min` (仅合法) |
| **Tree 输出** | `tree.predict()` | `tree.predict_proba()` + masking |
| **Action Masking** | ❌ 不支持 | ✅ 完全支持 |

### 6.2 DQN-VIPER 的 Criticality 计算

```python
# train/viper.py 第 196-203 行
if isinstance(model, DQN):
    obs_tensor = torch.from_numpy(obs).to(model.device)
    q_values = model.q_net(obs_tensor).detach().cpu().numpy()
    # q_values: (n_env, n_actions)
    return q_values.max(axis=1) - q_values.min(axis=1)  # 所有动作
```

**问题**: 对于 TicTacToe，包含非法动作的 Q 值。

### 6.3 MaskablePPO-VIPER 的改进

```python
# train/viper_maskable_ppo.py
log_probs = oracle.policy.get_distribution(obs).logits
for i in range(obs.shape[0]):
    legal_actions = np.where(obs[i] == 0)[0]
    legal_log_probs = log_probs[i][legal_actions]
    criticality = legal_log_probs.max() - legal_log_probs.min()  # 仅合法动作
```

**改进**: 只考虑合法动作，更准确。

### 6.4 何时使用哪个版本?

| 场景 | 推荐版本 |
|------|----------|
| CartPole, Pong (无 masking) | `viper.py` (DQN-VIPER) |
| TicTacToe (需要 masking) | `viper_maskable_ppo.py` ⭐ |
| 其他棋类游戏 (需要 masking) | `viper_maskable_ppo.py` |

---

## 7. 常见问题与解决方案

### Q1: `AttributeError: 'OrderEnforcing' object has no attribute 'board'`

**原因**: Gymnasium 的包装器层级太深，无法直接访问 `env.board`。

**解决方案**: 使用递归解包（已修复）

```python
def mask_fn(env):
    current_env = env
    while hasattr(current_env, 'env') and not hasattr(current_env, 'board'):
        current_env = current_env.env
    return (current_env.board == 0).astype(np.int8)
```

### Q2: VIPER 训练时只用一个对手，是否需要改成多对手?

**答案**: **不需要**。

**原因**:
1. VIPER 是模仿学习，不是从零学习
2. Oracle 已在多对手上训练
3. 单对手足以提取 Oracle 的知识
4. 符合 VIPER 原论文设计

**如果想要多对手**:
```python
# 修改 sample_trajectory_maskable_ppo
opponents = ['random', 'minmax']
for iteration in range(n_iter):
    opponent = random.choice(opponents)
    env = gym.make(env_name, opponent_type=opponent)
    # ...
```

### Q3: 决策树性能不如神经网络怎么办?

**答案**: 这是**正常的**。

**原因**:
- 决策树的表达能力 < 神经网络
- VIPER 的目标是**可解释性**，不是最高性能
- 通常可以达到 Oracle 的 70-90% 性能

**改进方法**:
1. 增加树的复杂度 (`--max-depth 12`, `--max-leaves 80`)
2. 增加采样数据 (`--total-timesteps 100000`)
3. 使用更强的 Oracle

### Q4: 如何验证实现是否正确?

**检查清单**:

1. ✅ **无非法动作**: 评估时 `illegal_moves == 0`
2. ✅ **Beta 调度**: 第一轮 beta=1，后续 beta=0
3. ✅ **数据累积**: Dataset 持续增长
4. ✅ **性能提升**: 后期 iteration 性能 > 初期
5. ✅ **与 Oracle 对比**: Tree 性能应达到 Oracle 的 70%+

**测试命令**:
```bash
# 训练
python train/viper_maskable_ppo.py \
    --oracle-path log/oracle_TicTacToe_ppo_aggressive.zip \
    --output log/viper_test.joblib \
    --total-timesteps 20000 \
    --n-iter 5 \
    --test

# 评估
python evaluation/evaluate_viper_tree.py \
    --model-path log/viper_test.joblib \
    --opponent both \
    --n-episodes 100
```

---

## 8. 总结

### 8.1 核心结论

1. **算法一致性**: 本实现与 VIPER 原论文的核心算法**完全一致**
2. **必要适配**: 针对 MaskablePPO 和 TicTacToe 做了**必要且合理**的适配
3. **单一对手**: 使用单一对手是**符合 VIPER 设计哲学**的，不是缺陷
4. **可解释性优先**: 选择单棵分类树优先保证**可解释性**

### 8.2 与原论文的对比总结

| 维度 | 一致性 | 说明 |
|------|--------|------|
| 核心算法流程 | ✅ 100% | DAgger + Weighted Imitation |
| Criticality Loss | ✅ 100% | log_prob_max - log_prob_min (仅合法动作) |
| Beta 调度 | ✅ 100% | β=1 then 0 |
| 树模型类型 | ✅ 100% | DecisionTreeClassifier |
| 样本加权 | ✅ 100% | sample_weight=criticality |
| 迭代选择 | ✅ 100% | 选择最佳树 |
| Oracle 类型 | 🔧 扩展 | 支持 MaskablePPO (原论文: DQN/PPO) |
| 非法动作处理 | 🔧 增强 | Probability masking (原论文: 未明确) |

### 8.3 推荐使用方式

```bash
# 标准训练（推荐）
python train/viper_maskable_ppo.py \
    --oracle-path log/oracle_TicTacToe_ppo_aggressive.zip \
    --output log/viper_TicTacToe_from_ppo.joblib \
    --total-timesteps 50000 \
    --n-iter 10 \
    --max-depth 10 \
    --max-leaves 50 \
    --opponent-type minmax \
    --test
```

---

## 参考文献

1. Bastani, O., et al. (2018). "Verifiable Reinforcement Learning via Policy Extraction." NeurIPS.
2. Ross, S., et al. (2011). "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning." AISTATS. (DAgger)
3. Haarnoja, T., et al. (2018). "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor." ICML. (Max-Entropy RL)
