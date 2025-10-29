# VIPER + MaskablePPO 学习指南

## 从零开始实现 VIPER 决策树训练

本指南将教你如何**从零开始**实现 VIPER 算法，帮助你深入理解每一步的原理和实现细节。

---

## 📚 需要的库

```python
# 核心库
import numpy as np                          # 数值计算
import torch                                # PyTorch（Oracle 使用）
import gymnasium as gym                     # 环境
from sklearn.tree import DecisionTreeClassifier  # 决策树
from sb3_contrib import MaskablePPO         # Oracle 模型
import joblib                               # 模型保存/加载
```

---

## 🎯 学习路线图

按照以下顺序实现，每一步都独立测试通过后再进行下一步。

---

## 步骤 1: 环境交互基础 🌍

### 目的
理解如何与 TicTacToe 环境交互，掌握基本的状态、动作、奖励概念。

### 需要实现

```python
def test_environment():
    """测试环境基本功能"""
    # 1. 创建环境
    # 2. Reset 获取初始状态
    # 3. 打印状态（9维向量）
    # 4. 尝试几个动作
    # 5. 观察 reward 和 done
```

### 参考文件
- **`gym_env/tictactoe.py`** - 环境完整实现
  - 重点：`reset()` 方法（第 75-83 行）
  - 重点：`step()` 方法（第 85-137 行）

### 关键概念

| 概念 | 说明 | 示例 |
|------|------|------|
| **observation** | 9维向量表示棋盘 | `[0, 0, 1, -1, 0, 0, 0, 0, 0]` |
| | `-1` = 对手的棋子 (O) | |
| | `0` = 空位 | |
| | `1` = 自己的棋子 (X) | |
| **action** | 0-8 的整数，表示位置 | `4` = 中心位置 |
| **reward** | +1 = 胜利 | |
| | -1 = 失败 | |
| | 0 = 平局 | |
| | -10 = 非法移动 | |
| **terminated** | 游戏是否结束 | `True` / `False` |

### 测试代码

```python
import gymnasium as gym
import gym_env  # 注册 TicTacToe 环境

# 创建环境
env = gym.make('TicTacToe-v0', opponent_type='random')

# Reset 获取初始状态
obs, info = env.reset()
print("初始状态:", obs)  # 应该是 [0, 0, 0, 0, 0, 0, 0, 0, 0]

# 下一步棋（选择中心位置）
action = 4
obs, reward, terminated, truncated, info = env.step(action)
print("新状态:", obs)     # 位置4应该变成1
print("奖励:", reward)
print("游戏结束:", terminated)

# 继续玩几步
done = terminated or truncated
while not done:
    # 随机选择合法动作
    legal_actions = [i for i in range(9) if obs[i] == 0]
    if not legal_actions:
        break
    action = np.random.choice(legal_actions)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(f"动作: {action}, 奖励: {reward}, 完成: {done}")

env.close()
```

### 验证清单
- [ ] 能成功创建环境
- [ ] Reset 返回全 0 的状态
- [ ] Step 能执行动作并返回 5 个值
- [ ] 非法动作会返回 -10 奖励
- [ ] 游戏能正常结束

---

## 步骤 2: 加载 Oracle 🧙

### 目的
加载你训练好的 MaskablePPO 模型，理解如何使用神经网络策略。

### 需要实现

```python
def load_oracle(oracle_path):
    """加载 MaskablePPO Oracle"""
    # 1. 创建环境
    # 2. 使用 MaskablePPO.load() 加载模型
    # 3. 返回 oracle 和 env
```

### 参考文件
- **`train/train_delta_selfplay_ppo.py`**
  - 重点：模型保存（第 314 行）
  - 加载是保存的反过程

### 关键 API

```python
from sb3_contrib import MaskablePPO
import gymnasium as gym

# 创建环境
env = gym.make('TicTacToe-v0', opponent_type='minmax')

# 加载模型
oracle = MaskablePPO.load('log/oracle_TicTacToe_ppo_aggressive.zip', env=env)

print("✓ Oracle 加载成功")
```

### Action Masking（重要！）

```python
# 获取当前状态的合法动作掩码
obs, _ = env.reset()
mask = (obs == 0).astype(bool)  # True=合法, False=非法

# 让 Oracle 选择动作（带 masking）
action, _ = oracle.predict(obs, action_masks=mask, deterministic=True)
print("Oracle 选择动作:", action)

# 验证动作合法
assert obs[action] == 0, "Oracle 选择了非法动作！"
```

### 测试代码

```python
def test_oracle():
    """测试 Oracle 加载和使用"""
    env = gym.make('TicTacToe-v0', opponent_type='minmax')
    oracle = MaskablePPO.load('log/oracle_TicTacToe_ppo_aggressive.zip', env=env)

    # 玩一局
    obs, _ = env.reset()
    done = False
    episode_reward = 0

    while not done:
        # 获取合法动作掩码
        mask = (obs == 0).astype(bool)

        # Oracle 选择动作
        action, _ = oracle.predict(obs, action_masks=mask, deterministic=True)

        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward

        print(f"动作: {action}, 奖励: {reward}")

    print(f"总奖励: {episode_reward}")
    env.close()
```

### 验证清单
- [ ] Oracle 能成功加载
- [ ] predict() 能返回动作
- [ ] 带 masking 时不会选择非法动作
- [ ] Oracle 能完整玩一局

---

## 步骤 3: 计算 Criticality（重要性）⚖️

### 目的
判断一个状态有多重要，为训练决策树提供样本权重。

### 数学原理

在 Max-Entropy RL 框架下：

```
Q(s,a) ≈ log π(a|s)

criticality(s) = max_a∈Legal Q(s,a) - min_a∈Legal Q(s,a)
               = max_a∈Legal log π(a|s) - min_a∈Legal log π(a|s)
```

**直观理解**:
- **高 criticality**: 选对动作很关键（例如：即将获胜的状态）
- **低 criticality**: 随便选都差不多（例如：开局第一步）

### 需要实现

```python
def compute_criticality(oracle, observation):
    """计算状态的 criticality"""
    # 1. 将 observation 转为 tensor
    # 2. 使用 oracle.policy.get_distribution() 获取分布
    # 3. 提取 log probabilities (logits)
    # 4. 找到合法动作（observation == 0）
    # 5. 计算合法动作的 max - min
    # 6. 返回 criticality（float）
```

### 参考文件
- **`train/viper.py`** 第 187-222 行 `get_loss()` 函数
  - 重点：PPO 部分（第 204-220 行）

### 关键 API

```python
import torch

def compute_criticality(oracle, observation):
    """计算 criticality"""
    with torch.no_grad():
        # 1. 转为 tensor
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32)
        obs_tensor = obs_tensor.unsqueeze(0)  # (1, 9)
        obs_tensor = obs_tensor.to(oracle.device)

        # 2. 获取策略分布
        distribution = oracle.policy.get_distribution(obs_tensor)
        log_probs = distribution.distribution.logits.cpu().numpy()[0]  # (9,)

        # 3. 找合法动作
        legal_actions = np.where(observation == 0)[0]

        if len(legal_actions) == 0:
            return 0.0  # 无合法动作（不应该发生）

        # 4. 计算合法动作的 Q 值范围
        legal_log_probs = log_probs[legal_actions]
        criticality = legal_log_probs.max() - legal_log_probs.min()

        return float(criticality)
```

### 示例

```python
# 状态1: 即将获胜（对手有两个连成一线，我必须堵）
obs1 = np.array([0, -1, -1,  # O O .
                 0,  1,  0,  # . X .
                 0,  0,  0]) # . . .
legal_actions = [0, 3, 5, 6, 7, 8]
log_probs = {0: -0.1, 3: -3.0, 5: -2.5, ...}  # 位置0几乎必选
criticality1 = -0.1 - (-3.0) = 2.9  # 很重要！

# 状态2: 开局第一步
obs2 = np.array([0, 0, 0,
                 0, 0, 0,
                 0, 0, 0])
legal_actions = [0,1,2,3,4,5,6,7,8]
log_probs = [-1.2, -1.3, -1.1, -1.2, -1.0, ...]  # 差不多
criticality2 = -1.0 - (-1.3) = 0.3  # 不重要
```

### 测试代码

```python
def test_criticality():
    """测试 criticality 计算"""
    env = gym.make('TicTacToe-v0', opponent_type='minmax')
    oracle = MaskablePPO.load('log/oracle_TicTacToe_ppo_aggressive.zip', env=env)

    # 测试几个状态
    test_states = [
        # 开局
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        # 中局
        np.array([1, -1, 0, 0, 1, 0, 0, 0, -1], dtype=np.float32),
        # 关键时刻（两个X连一线）
        np.array([1, 1, 0, -1, 0, 0, 0, -1, 0], dtype=np.float32),
    ]

    for i, obs in enumerate(test_states):
        crit = compute_criticality(oracle, obs)
        print(f"状态{i+1} criticality: {crit:.3f}")
```

### 验证清单
- [ ] 函数能返回数值（不报错）
- [ ] Criticality 在合理范围（通常 0-5）
- [ ] 关键状态的 criticality > 普通状态
- [ ] 没有合法动作时返回 0

---

## 步骤 4: 采样轨迹 🎲

### 目的
在环境中采样，收集训练数据 `(state, action, weight)`。

### VIPER 数据收集策略

**第一轮（iteration 0）**:
- 使用 **Oracle** 采样
- 目的：收集高质量初始数据

**后续轮（iteration 1+）**:
- 使用 **Tree** 采样（DAgger 策略）
- 目的：让 Tree 在自己会遇到的状态上学习
- 修正协变量偏移（Covariate Shift）

### 需要实现

```python
def sample_trajectories(oracle, env, n_steps, use_oracle=True):
    """采样 n_steps 个样本

    Returns:
        dataset: List of (observation, action, weight)
    """
    dataset = []
    obs, _ = env.reset()

    while len(dataset) < n_steps:
        # 1. 选择动作（Oracle 或其他策略）
        if use_oracle:
            # 使用 Oracle
            mask = (obs == 0).astype(bool)
            action, _ = oracle.predict(obs, action_masks=mask, deterministic=True)
        else:
            # 使用 Tree（或随机）
            legal_actions = np.where(obs == 0)[0]
            action = np.random.choice(legal_actions)

        # 2. 获取 Oracle 的动作（作为标签）
        mask = (obs == 0).astype(bool)
        oracle_action, _ = oracle.predict(obs, action_masks=mask, deterministic=True)

        # 3. 计算 criticality（权重）
        criticality = compute_criticality(oracle, obs)

        # 4. 保存样本
        dataset.append((obs.copy(), oracle_action, criticality))

        # 5. 执行动作，获取下一个状态
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs

    return dataset
```

### 参考文件
- **`train/viper.py`** 第 137-184 行 `sample_trajectory()` 函数

### 数据格式

```python
# dataset 是一个列表，每个元素是三元组
dataset = [
    (obs1, action1, weight1),  # 样本1
    (obs2, action2, weight2),  # 样本2
    ...
]

# 示例
obs = np.array([0, 0, 1, -1, 0, 0, 0, 0, 0])  # 状态
action = 4                                      # Oracle 选择的动作
weight = 1.5                                    # Criticality
```

### 测试代码

```python
def test_sampling():
    """测试采样"""
    env = gym.make('TicTacToe-v0', opponent_type='minmax')
    oracle = MaskablePPO.load('log/oracle_TicTacToe_ppo_aggressive.zip', env=env)

    # 采样 100 个样本
    dataset = sample_trajectories(oracle, env, n_steps=100, use_oracle=True)

    print(f"✓ 采样完成，收集 {len(dataset)} 个样本")

    # 检查数据格式
    obs, action, weight = dataset[0]
    print(f"  状态 shape: {obs.shape}")      # (9,)
    print(f"  动作: {action}")                # 0-8
    print(f"  权重: {weight:.3f}")            # float

    # 检查合法性
    for obs, action, weight in dataset[:10]:
        assert obs[action] == 0, f"非法动作！obs[{action}] = {obs[action]}"
    print("✓ 所有动作合法")
```

### 验证清单
- [ ] 能采样指定数量的样本
- [ ] 每个样本是 (obs, action, weight) 三元组
- [ ] 所有动作都是合法的（obs[action] == 0）
- [ ] Weight 在合理范围内

---

## 步骤 5: 训练决策树 🌳

### 目的
使用 sklearn 训练决策树，模仿 Oracle 的行为。

### 需要实现

```python
def train_decision_tree(dataset, max_depth=10, max_leaves=50):
    """训练决策树

    Args:
        dataset: [(obs, action, weight), ...]
        max_depth: 树的最大深度
        max_leaves: 最大叶子节点数

    Returns:
        tree: 训练好的 DecisionTreeClassifier
    """
    # 1. 准备训练数据
    X = np.array([sample[0] for sample in dataset])      # (N, 9)
    y = np.array([sample[1] for sample in dataset])      # (N,)
    weights = np.array([sample[2] for sample in dataset])# (N,)

    # 2. 创建决策树
    tree = DecisionTreeClassifier(
        criterion='entropy',        # 使用信息增益
        max_depth=max_depth,        # 限制深度（控制可解释性）
        max_leaf_nodes=max_leaves,  # 限制叶子数
        random_state=42,
        ccp_alpha=0.0001           # 剪枝参数（防止过拟合）
    )

    # 3. 训练（带权重）
    tree.fit(X, y, sample_weight=weights)

    print(f"✓ 训练完成")
    print(f"  树深度: {tree.tree_.max_depth}")
    print(f"  叶子节点数: {tree.tree_.n_leaves}")

    return tree
```

### 参考文件
- **`train/viper.py`** 第 75-84 行
- sklearn 文档: https://scikit-learn.org/stable/modules/tree.html

### 关键参数说明

| 参数 | 作用 | 推荐值 |
|------|------|--------|
| `criterion='entropy'` | 分裂准则（信息增益） | entropy |
| `max_depth` | 树深度（越深越复杂） | 8-12 |
| `max_leaf_nodes` | 叶子数（控制大小） | 30-80 |
| `ccp_alpha` | 剪枝强度（防过拟合） | 0.0001 |
| `sample_weight` | 样本权重（criticality） | 必须提供 |

### 为什么需要 sample_weight？

```python
# 不使用权重（普通训练）
tree.fit(X, y)
# 所有样本平等对待，开局和决胜时刻一样重要 ❌

# 使用权重（VIPER 的核心）
tree.fit(X, y, sample_weight=weights)
# 重要状态权重高，Tree 优先学好关键决策 ✅
```

### 测试代码

```python
def test_tree_training():
    """测试决策树训练"""
    env = gym.make('TicTacToe-v0', opponent_type='minmax')
    oracle = MaskablePPO.load('log/oracle_TicTacToe_ppo_aggressive.zip', env=env)

    # 采样数据
    dataset = sample_trajectories(oracle, env, n_steps=1000, use_oracle=True)

    # 训练树
    tree = train_decision_tree(dataset, max_depth=8, max_leaves=30)

    # 测试预测
    obs = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)
    action = tree.predict(obs.reshape(1, -1))[0]
    print(f"树预测动作: {action}")
```

### 验证清单
- [ ] 树能成功训练（不报错）
- [ ] 树的深度和叶子数在预期范围
- [ ] 树能预测动作（0-8）
- [ ] 不同数据集训练出不同的树

---

## 步骤 6: 决策树推理（带 Masking）🎭

### 目的
让决策树在推理时保证选择合法动作。

### 核心思想

**训练时**: 直接学习 Oracle 的动作标签（可能训练数据中有非法动作的样本）

**推理时**: 应用 masking，只选择合法动作

```python
# 训练: 学习 Oracle 的标签
tree.fit(X, y)  # y 可能包含任何动作 0-8

# 推理: 强制选合法动作
probs = tree.predict_proba(obs)      # 所有动作的概率
legal_probs = probs[legal_actions]    # 只看合法动作
action = legal_actions[argmax(legal_probs)]  # 选最好的合法动作
```

### 需要实现

```python
class TreePolicy:
    """决策树策略包装器"""

    def __init__(self, tree):
        self.tree = tree
        self.n_actions = 9  # TicTacToe

    def predict(self, observation, deterministic=True):
        """预测动作（带 masking）"""
        # 处理输入 shape
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
            single_obs = True
        else:
            single_obs = False

        # 获取概率分布
        probs = self.tree.predict_proba(observation)  # (batch, n_classes)

        actions = []
        for i in range(observation.shape[0]):
            obs = observation[i]
            prob = probs[i]

            # 获取合法动作
            legal_actions = np.where(obs == 0)[0]

            if len(legal_actions) == 0:
                # 无合法动作（不应该发生）
                actions.append(0)
                continue

            # 只考虑合法动作的概率
            legal_probs = prob[legal_actions]

            # 选择概率最高的合法动作
            best_idx = np.argmax(legal_probs)
            action = legal_actions[best_idx]

            actions.append(action)

        actions = np.array(actions)

        if single_obs:
            return actions[0], None
        else:
            return actions, None
```

### 参考文件
- **`train/viper_single_tree.py`** 第 30-97 行 `ProbabilityMaskedTreeWrapper`

### 为什么这样设计？

| 方案 | 优点 | 缺点 |
|------|------|------|
| **训练时过滤非法动作** | 训练数据干净 | 损失信息，数据量减少 |
| **推理时 masking**（本方案）✅ | 保留所有数据 | 需要额外包装器 |

### 测试代码

```python
def test_tree_policy():
    """测试 TreePolicy 的 masking"""
    env = gym.make('TicTacToe-v0', opponent_type='minmax')
    oracle = MaskablePPO.load('log/oracle_TicTacToe_ppo_aggressive.zip', env=env)

    # 训练一个简单的树
    dataset = sample_trajectories(oracle, env, n_steps=1000, use_oracle=True)
    tree = train_decision_tree(dataset, max_depth=8, max_leaves=30)

    # 包装成 Policy
    policy = TreePolicy(tree)

    # 测试 100 个状态，检查是否有非法动作
    illegal_count = 0
    for _ in range(100):
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = policy.predict(obs)

            # 检查合法性
            if obs[action] != 0:
                illegal_count += 1
                print(f"❌ 非法动作！obs[{action}] = {obs[action]}")

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    print(f"✓ 测试完成，非法动作数: {illegal_count}")
    assert illegal_count == 0, "存在非法动作！"
```

### 验证清单
- [ ] TreePolicy 能成功预测动作
- [ ] 所有预测的动作都是合法的
- [ ] 能处理单个和批量观察
- [ ] 无合法动作时有合理的兜底

---

## 步骤 7: 评估策略 📊

### 目的
测试决策树对战不同对手的性能。

### 需要实现

```python
def evaluate_policy(policy, env_name='TicTacToe-v0',
                   opponent_type='minmax', n_episodes=100):
    """评估策略

    Returns:
        结果字典，包含 mean_reward, win_rate 等
    """
    env = gym.make(env_name, opponent_type=opponent_type)

    episode_rewards = []
    wins, draws, losses = 0, 0, 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            # 使用策略选择动作
            action, _ = policy.predict(obs, deterministic=True)

            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        episode_rewards.append(episode_reward)

        # 统计胜负平
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1

    env.close()

    # 计算统计量
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'win_rate': wins / n_episodes,
        'draw_rate': draws / n_episodes,
        'loss_rate': losses / n_episodes,
        'wins': wins,
        'draws': draws,
        'losses': losses
    }
```

### 性能评估标准

#### 对战 MinMax（最优对手）

| 性能等级 | 平局率 | 说明 |
|---------|--------|------|
| ✅ 优秀 | ≥ 80% | 学到了接近最优策略 |
| △ 良好 | 60-80% | 还有提升空间 |
| ✗ 需改进 | < 60% | 需要更多训练 |

**为什么看平局率？**
- MinMax 是最优策略，双方都不会输
- 高平局率说明 Tree 也接近最优

#### 对战 Random

| 性能等级 | 胜率 | 说明 |
|---------|------|------|
| ✅ 优秀 | ≥ 90% | 能稳定战胜弱对手 |
| △ 良好 | 70-90% | 基本掌握 |
| ✗ 需改进 | < 70% | 策略有问题 |

### 测试代码

```python
def test_evaluation():
    """测试评估"""
    env = gym.make('TicTacToe-v0', opponent_type='minmax')
    oracle = MaskablePPO.load('log/oracle_TicTacToe_ppo_aggressive.zip', env=env)

    # 训练一个树
    dataset = sample_trajectories(oracle, env, n_steps=5000, use_oracle=True)
    tree = train_decision_tree(dataset, max_depth=10, max_leaves=50)
    policy = TreePolicy(tree)

    # 评估
    print("\n评估结果:")
    for opponent in ['random', 'minmax']:
        results = evaluate_policy(policy, opponent_type=opponent, n_episodes=100)
        print(f"\nvs {opponent.upper()}:")
        print(f"  胜率: {results['win_rate']*100:.1f}%")
        print(f"  平局率: {results['draw_rate']*100:.1f}%")
        print(f"  负率: {results['loss_rate']*100:.1f}%")
        print(f"  平均奖励: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
```

### 验证清单
- [ ] 能评估 100 局
- [ ] 返回正确的统计信息
- [ ] 胜/平/负总和 = 100
- [ ] 对战 Random 胜率 > 50%

---

## 步骤 8: VIPER 主循环 🔄

### 目的
将所有步骤串联起来，实现完整的 VIPER 算法。

### VIPER 算法流程

```
输入:
  - oracle: 预训练的 MaskablePPO
  - n_iterations: 迭代次数（例如 10）
  - samples_per_iter: 每轮采样数（例如 5000）

流程:
  1. 加载 Oracle
  2. 初始化空数据集 D = []
  3. FOR i = 1 to n_iterations:
      a. 采样轨迹（第1轮用Oracle，后续用Tree）
      b. 聚合数据: D = D ∪ new_data
      c. 训练决策树
      d. 评估性能
      e. 记录当前树
  4. 返回性能最好的树
```

### 需要实现

```python
def train_viper(oracle_path, output_path,
                n_iterations=10, samples_per_iter=5000,
                max_depth=10, max_leaves=50):
    """VIPER 主训练流程"""

    print("="*70)
    print("VIPER 训练开始")
    print("="*70)

    # 步骤 1: 加载 Oracle
    env = gym.make('TicTacToe-v0', opponent_type='minmax')
    oracle = MaskablePPO.load(oracle_path, env=env)
    print("✓ Oracle 加载成功")

    # 步骤 2: 初始化
    all_data = []      # 累积所有数据（DAgger 的核心）
    all_trees = []     # 保存所有训练的树
    all_rewards = []   # 记录每棵树的性能

    # 步骤 3: VIPER 迭代
    for iteration in range(n_iterations):
        print(f"\n{'='*70}")
        print(f"Iteration {iteration + 1}/{n_iterations}")
        print(f"{'='*70}")

        # 3a. 采样轨迹
        # 第一轮用 Oracle，后续可以用 Tree（这里简化始终用 Oracle）
        use_oracle = True  # 或者: (iteration == 0)
        new_data = sample_trajectories(oracle, env, samples_per_iter, use_oracle)

        # 3b. 聚合数据（DAgger 的关键：累积数据）
        all_data.extend(new_data)
        print(f"✓ 累积数据集大小: {len(all_data)}")

        # 3c. 训练决策树
        tree = train_decision_tree(all_data, max_depth, max_leaves)
        all_trees.append(tree)

        # 3d. 评估性能
        policy = TreePolicy(tree)
        results = evaluate_policy(policy, opponent_type='minmax', n_episodes=100)
        all_rewards.append(results['mean_reward'])

        print(f"✓ 评估结果:")
        print(f"  平均奖励: {results['mean_reward']:.3f}")
        print(f"  胜: {results['wins']}, 平: {results['draws']}, 负: {results['losses']}")

    # 步骤 4: 选择最佳树
    print("\n" + "="*70)
    print("VIPER 训练完成")
    print("="*70)

    best_idx = np.argmax(all_rewards)
    best_tree = all_trees[best_idx]
    best_reward = all_rewards[best_idx]

    print(f"最佳树: Iteration {best_idx + 1}")
    print(f"最佳奖励: {best_reward:.3f}")

    # 保存
    best_policy = TreePolicy(best_tree)
    joblib.dump(best_tree, output_path)
    print(f"✓ 模型保存到: {output_path}")

    # 最终测试
    print("\n最终测试:")
    for opponent in ['random', 'minmax']:
        results = evaluate_policy(best_policy, opponent_type=opponent, n_episodes=100)
        print(f"\nvs {opponent.upper()}:")
        print(f"  胜率: {results['win_rate']*100:.1f}%")
        print(f"  平局率: {results['draw_rate']*100:.1f}%")

    return best_policy
```

### 参考文件
- **`train/viper.py`** 第 63-100 行 `train_viper()` 函数

### DAgger 数据聚合的重要性

```python
# ❌ 错误方式：每轮丢弃旧数据
for i in range(n_iterations):
    data = sample(...)  # 只用新数据
    tree.fit(data)      # 数据量不增长

# ✅ 正确方式：累积数据（DAgger）
all_data = []
for i in range(n_iterations):
    new_data = sample(...)
    all_data.extend(new_data)  # 累积！
    tree.fit(all_data)          # 数据量持续增长
```

**为什么累积？**
- 第 1 轮：5000 样本
- 第 2 轮：10000 样本（包含前面的）
- 第 10 轮：50000 样本
- 更多数据 → 更好的树

### 完整运行示例

```python
if __name__ == '__main__':
    train_viper(
        oracle_path='log/oracle_TicTacToe_ppo_aggressive.zip',
        output_path='log/viper_my_tree.joblib',
        n_iterations=10,
        samples_per_iter=5000,
        max_depth=10,
        max_leaves=50
    )
```

### 验证清单
- [ ] 能完整运行 10 轮
- [ ] 数据集持续增长
- [ ] 每轮都能评估性能
- [ ] 最后选出最佳树并保存
- [ ] 最终测试显示合理性能

---

## 🔑 核心概念总结

### 1. VIPER = Imitation Learning + DAgger

| 组件 | 角色 | 说明 |
|------|------|------|
| **Oracle** | 教师 | 高性能神经网络（MaskablePPO） |
| **Tree** | 学生 | 可解释决策树 |
| **Imitation** | 方法 | Tree 学习"Oracle 会怎么做" |
| **DAgger** | 技巧 | 数据聚合，修正分布偏移 |

### 2. Criticality（状态重要性）

```
criticality(s) = max_a Q(s,a) - min_a Q(s,a)

高 criticality → 重要状态 → 训练权重高 → Tree 优先学好
低 criticality → 普通状态 → 训练权重低 → 可以容忍错误
```

### 3. Action Masking（保证合法）

```
训练: 学习 Oracle 的动作标签（可能有非法动作样本）
推理: 应用 masking，只选合法动作（100% 合法）
```

### 4. 为什么需要 DAgger？

**问题**: 普通模仿学习的协变量偏移

```
Oracle 采样 → Tree 学习 → Tree 犯错 → 进入新状态 → Oracle 没见过 → Tree 不知道怎么办 → 性能崩溃
```

**解决**: DAgger 数据聚合

```
第1轮: Oracle 采样 → Tree 学习
第2轮: Tree 采样（自己的状态分布）→ Oracle 标注 → Tree 在自己的错误上学习
...
```

---

## 📖 推荐学习顺序

1. ✅ **环境交互**（步骤1）- 最基础，必须先熟悉
2. ✅ **加载 Oracle**（步骤2）- 理解如何使用神经网络
3. ✅ **计算 Criticality**（步骤3）- 理解状态重要性
4. ✅ **采样数据**（步骤4）- 数据收集是核心
5. ✅ **训练树**（步骤5）- 最简单，调用 sklearn
6. ✅ **树推理 + Masking**（步骤6）- 关键！保证合法
7. ✅ **评估**（步骤7）- 验证效果
8. ✅ **主循环**（步骤8）- 串联所有步骤

---

## 📂 重要参考文件索引

| 文件 | 关键内容 | 重点行数 |
|------|----------|----------|
| **`train/viper.py`** | 完整 VIPER 实现 | 63-100, 137-184, 187-222 |
| **`train/viper_single_tree.py`** | 单树 + Masking | 30-97, 125-201 |
| **`gym_env/tictactoe.py`** | TicTacToe 环境 | 75-137 |
| **`train/train_delta_selfplay_ppo.py`** | MaskablePPO 使用 | 全文 |
| **sklearn 文档** | DecisionTreeClassifier | - |

---

## 🐛 调试技巧

### 1. 打印一切
```python
print(f"obs shape: {obs.shape}")
print(f"action: {action}, legal: {obs[action] == 0}")
print(f"criticality: {crit:.3f}")
```

### 2. 从小数据开始
```python
# 先测试 100 个样本
dataset = sample_trajectories(oracle, env, n_steps=100)

# 确认没问题后再用 5000
dataset = sample_trajectories(oracle, env, n_steps=5000)
```

### 3. 检查合法性
```python
# 在评估时统计非法动作
illegal_count = 0
for episode in range(100):
    # ...
    if obs[action] != 0:
        illegal_count += 1

assert illegal_count == 0, "有非法动作！"
```

### 4. 可视化决策
```python
# 打印几个状态和 Tree 的选择
def visualize_decision(policy, obs):
    print("棋盘:")
    for i in range(3):
        row = obs[i*3:(i+1)*3]
        print(' '.join(['.XO'[int(x)+1] for x in row]))

    action, _ = policy.predict(obs)
    print(f"Tree 选择: {action}")
```

---

## 💡 快速验证清单

在开始写代码前，确认以下准备工作：

- [ ] 已安装所有必要的库
- [ ] 已有训练好的 Oracle 模型（.zip 文件）
- [ ] TicTacToe 环境可以正常运行
- [ ] 理解了 VIPER 的基本原理

在完成代码后，验证以下功能：

- [ ] 环境能 reset 和 step
- [ ] Oracle 能加载和 predict
- [ ] Criticality 返回合理数值（0-5）
- [ ] 采样能收集到数据
- [ ] 树能训练（无报错）
- [ ] 树推理时没有非法动作
- [ ] 评估能统计胜负平
- [ ] 主循环能跑完 10 轮

---

## 🎓 进阶话题

完成基础实现后，可以尝试：

### 1. 真正的 DAgger
```python
# 第 1 轮用 Oracle，后续用 Tree
use_oracle = (iteration == 0)
if use_oracle:
    action = oracle.predict(...)
else:
    action = tree_policy.predict(...)  # 用当前 Tree 采样
```

### 2. 可视化决策树
```python
from sklearn.tree import export_text

feature_names = [f"pos_{i}" for i in range(9)]
rules = export_text(tree, feature_names=feature_names)
print(rules)
```

### 3. 规则提取
```python
from sklearn.tree import export_text

# 导出可读规则
rules = export_text(tree,
                   feature_names=[f"pos_{i}" for i in range(9)],
                   class_names=[str(i) for i in range(9)])
with open('tree_rules.txt', 'w') as f:
    f.write(rules)
```

### 4. 对比 Oracle vs Tree
```python
# 评估 Oracle
oracle_policy = OracleWrapper(oracle)
oracle_results = evaluate_policy(oracle_policy, n_episodes=100)

# 评估 Tree
tree_results = evaluate_policy(tree_policy, n_episodes=100)

# 对比
print(f"Oracle 平局率: {oracle_results['draw_rate']*100:.1f}%")
print(f"Tree 平局率: {tree_results['draw_rate']*100:.1f}%")
```

---

## 📚 相关文档

- **[VIPER_MASKABLE_PPO_GUIDE.md](VIPER_MASKABLE_PPO_GUIDE.md)** - 完整使用指南
- **[VIPER_TECHNICAL_ANALYSIS.md](VIPER_TECHNICAL_ANALYSIS.md)** - 技术深度分析
- **[VIPER_QUICK_REFERENCE.md](VIPER_QUICK_REFERENCE.md)** - 快速参考

---

## 🎉 完成后

恭喜！完成所有步骤后，你将拥有：

✅ 深入理解 VIPER 算法的每一步
✅ 一个可解释的决策树策略
✅ 能够提取和理解决策规则
✅ 掌握模仿学习和 DAgger 的核心思想

现在开始动手吧！遇到问题随时查阅本指南。🚀
