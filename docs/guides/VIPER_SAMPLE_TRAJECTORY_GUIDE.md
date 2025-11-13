# VIPER 采样轨迹详解：从原理到实现

## 📌 核心概念

在实现`sample_trajectory`之前，**必须理解**这三个关键概念：

### 1. 混合策略采样（Beta参数）🎲

```python
# VIPER使用beta控制"谁来执行动作"
active_policy = [policy, oracle][np.random.binomial(1, beta)]
```

**为什么要混合？**

| 策略 | 能探索到的状态 | 问题 |
|------|--------------|------|
| **100% Oracle** | Oracle的轨迹 | Tree只见过Oracle的状态，自己犯错后不知道怎么办 ❌ |
| **100% Tree** | Tree的轨迹 | Tree一直犯同样的错，没机会学到正确行为 ❌ |
| **混合 (Beta)** | 两者都有 | 平衡探索和学习 ✅ |

**Beta的演化**:
```
第1轮: beta=1.0  → 100%用Oracle → 收集高质量初始数据
第2轮: beta=0.5  → 50%用Tree    → Tree开始探索自己的错误
第3轮: beta=0.25 → 75%用Tree    → 更多让Tree自己走
...
```

### 2. Oracle标签（模仿学习的本质）🎯

**极其重要**：无论**谁执行动作**，都记录**Oracle会做什么**！

```python
# 执行动作的策略
active_policy = [tree, oracle][use_oracle_for_action]
action = active_policy.predict(...)  # 这个动作用来env.step()

# 但是！标签永远是Oracle的动作
oracle_action = oracle.predict(...)  # 这个动作用来训练Tree
dataset.append((obs, oracle_action, weight))  # 注意：是oracle_action！
```

**为什么这样设计？**

```
场景：Tree走到一个糟糕的状态（即将输棋）

错误做法：
  dataset.append((obs, tree_action, weight))  # ❌ 学习Tree的错误

正确做法：
  dataset.append((obs, oracle_action, weight))  # ✅ 学Oracle的补救

结果：Tree学会"在糟糕状态下Oracle会如何补救"
```

### 3. 谁执行 vs 谁标注 📝

| 步骤 | 使用的策略 | 作用 |
|------|----------|------|
| **选择动作执行** | `active_policy` (Tree或Oracle) | 决定**进入哪些状态** |
| **生成训练标签** | **永远是Oracle** | 决定**学什么行为** |

---

## 📖 VIPER原始代码详解

让我们逐行分析`train/viper.py`中的`sample_trajectory`函数：

```python
def sample_trajectory(args, policy, beta):
    # policy: 当前的决策树（可能为None）
    # beta: Oracle使用概率

    # 为每轮VIPER创建新环境（避免环境状态污染）
    env, oracle = load_oracle_env(args)
    policy = policy or oracle  # 如果没有policy（第1轮），用oracle

    trajectory = []
    obs = env.reset()
    n_steps = args.total_timesteps // args.n_iter

    while len(trajectory) < n_steps:
        # 🔑 关键1：混合策略 - 决定谁来执行动作
        active_policy = [policy, oracle][np.random.binomial(1, beta)]
        #                ^^^^^^  ^^^^^^
        #                 Tree   Oracle
        # beta=1.0 → 100%选Oracle
        # beta=0.5 → 50%选Oracle, 50%选Tree

        # 根据active_policy的类型选择动作
        if isinstance(active_policy, DecisionTreeClassifier):
            # Pong需要提取特征
            if is_pong:
                extracted_obs = env.get_extracted_obs()
                action = active_policy.predict(extracted_obs)
            else:
                action = active_policy.predict(obs)
        else:
            # Oracle预测（神经网络）
            action, _states = active_policy.predict(obs, deterministic=True)

        # 🔑 关键2：获取Oracle的标签
        # 注意：这里再次调用oracle，无论active_policy是谁！
        if not isinstance(active_policy, DecisionTreeClassifier):
            # 如果active_policy本身就是oracle，复用action
            oracle_action = action
        else:
            # 如果active_policy是tree，重新调用oracle获取标签
            oracle_action = oracle.predict(obs, deterministic=True)[0]

        # 执行动作（使用active_policy选的动作，不是oracle_action！）
        step_result = env.step(action)
        if len(step_result) == 5:
            next_obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            next_obs, reward, done, info = step_result

        # 🔑 关键3：计算状态重要度
        state_loss = get_loss(env, oracle, obs)

        # 🔑 关键4：记录(状态, Oracle的动作, 权重)
        if is_pong:
            extracted_obs = env.get_extracted_obs()
            trajectory += list(zip(extracted_obs, oracle_action, state_loss))
        else:
            trajectory += list(zip(obs, oracle_action, state_loss))
        #                           ^^^  ^^^^^^^^^^^^  ^^^^^^^^^^
        #                           状态  Oracle标签    Criticality

        obs = next_obs

        if done:
            obs = env.reset()

    return trajectory
```

**关键代码分析**：

```python
# 第42-43行：混合策略
active_policy = [policy, oracle][np.random.binomial(1, beta)]

# np.random.binomial(1, beta) 返回 0 或 1
# - 返回1的概率 = beta
# - 返回0的概率 = 1-beta
#
# beta=1.0 → 100%返回1 → 100%选oracle
# beta=0.5 → 50%返回1  → 50%选oracle, 50%选policy
# beta=0.0 → 0%返回1   → 100%选policy

# 第53-58行：获取Oracle标签
if not isinstance(active_policy, DecisionTreeClassifier):
    oracle_action = action  # active_policy就是oracle，复用
else:
    oracle_action = oracle.predict(obs, deterministic=True)[0]  # 重新调用

# 为什么要这样判断？
# - 如果active_policy是oracle，action已经是oracle的选择，不用再调用
# - 如果active_policy是tree，必须调用oracle获取标签

# 第68行：执行动作
env.step(action)  # 用的是active_policy的动作，不是oracle_action

# 第73行：保存样本
trajectory += list(zip(obs, oracle_action, state_loss))
# 保存的是oracle_action！
```

---

## 🎯 Mask PPO适配版本

### 关键区别

TicTacToe与Atari游戏的核心差异：

| 环境 | 动作空间 | 挑战 |
|------|---------|------|
| **Atari (Pong)** | 所有动作随时可用 | 无需masking |
| **TicTacToe** | 只能下空位 | **必须masking** |

### 实现框架

```python
def sample_trajectory(oracle, policy, env, n_steps, beta=0.5, use_criticality=True):
    """采样 n_steps 个样本（Mask PPO版本）

    VIPER算法核心思想：
    1. 混合策略采样：用beta控制使用oracle vs policy来执行动作
       - beta越大，越多用oracle执行（探索oracle的轨迹）
       - beta越小，越多用policy执行（探索policy的错误状态）
    2. Oracle标签：无论用哪个策略执行，都记录oracle会做什么
       - 这是模仿学习的关键：学习在各种状态下oracle的行为
    3. 状态重要度：用criticality给状态加权
       - 重要的决策点权重更高

    Args:
        oracle: 专家策略 (MaskablePPO)
        policy: 当前策略 (DecisionTreeClassifier or None)
        env: TicTacToe环境
        n_steps: 采样步数
        beta: oracle使用概率 (0.0-1.0)
              - 1.0 = 完全使用oracle
              - 0.5 = 50%使用oracle, 50%使用policy
              - 0.0 = 完全使用policy
        use_criticality: 是否计算状态重要度权重

    Returns:
        trajectory: List of (observation, oracle_action, weight)
    """

    # ========== TODO 1: 初始化 ==========
    # 1.1 创建空轨迹列表
    # 1.2 如果policy是None，设为oracle（第一轮迭代）
    # 1.3 Reset环境，获取初始观察

    # ========== TODO 2: 采样循环 ==========
    # while len(trajectory) < n_steps:

        # ----- TODO 2.1: 混合策略 - 决定谁执行动作 -----
        # 提示：
        # use_oracle_for_action = np.random.binomial(1, beta) == 1
        # active_policy = oracle if use_oracle_for_action else policy
        #
        # 理解：
        # - np.random.binomial(1, beta) 返回0或1，1的概率为beta
        # - 如果为True，用oracle；否则用policy

        # ----- TODO 2.2: 获取Action Mask（TicTacToe特有！）-----
        # 提示：
        # mask = (obs == 0).astype(bool)
        # mask_tensor = torch.tensor(mask).unsqueeze(0)
        #
        # 解释：
        # - obs == 0 找到所有空位
        # - True表示该位置合法，False表示已占用

        # ----- TODO 2.3: 使用active_policy选择要执行的动作 -----
        # 如果active_policy是DecisionTreeClassifier：
        #     action = policy.predict(obs.reshape(1, -1))[0]
        #
        #     # 检查合法性（决策树可能预测非法动作）
        #     if not mask[action]:
        #         # 预测了非法动作，随机选择一个合法动作
        #         valid_actions = np.where(mask)[0]
        #         action = np.random.choice(valid_actions)
        #
        # 如果active_policy是MaskablePPO：
        #     action, _ = active_policy.predict(
        #         obs,
        #         action_masks=mask_tensor,  # MaskablePPO会自动处理
        #         deterministic=True
        #     )

        # ----- TODO 2.4: 获取Oracle的动作作为标签（关键！）-----
        # 提示：
        # oracle_action, _ = oracle.predict(
        #     obs,
        #     action_masks=mask_tensor,
        #     deterministic=True
        # )
        #
        # ⚠️ 注意：
        # - 无论active_policy是谁，都要调用这个！
        # - 这是VIPER的核心：标签永远来自Oracle
        # - 即使active_policy是oracle本身，也要调用（虽然结果一样）
        #   或者可以优化：
        #   if use_oracle_for_action:
        #       oracle_action = action  # 复用
        #   else:
        #       oracle_action, _ = oracle.predict(...)

        # ----- TODO 2.5: 计算criticality（权重）-----
        # if use_criticality:
        #     try:
        #         weight = compute_criticality(oracle, obs)[0]
        #     except Exception as e:
        #         print(f"Warning: criticality计算失败: {e}")
        #         weight = 1.0  # 失败时用默认权重
        # else:
        #     weight = 1.0  # 均匀权重
        #
        # 解释：
        # - criticality高的状态更重要，在训练时权重更大
        # - 如果不用criticality，所有状态权重相同

        # ----- TODO 2.6: 保存样本 -----
        # trajectory.append((obs.copy(), oracle_action, weight))
        #
        # ⚠️ 注意三个关键点：
        # 1. obs.copy() - 避免引用问题，必须复制
        # 2. oracle_action - 不是action！标签来自Oracle
        # 3. weight - criticality权重

        # ----- TODO 2.7: 执行动作，获取下一个状态 -----
        # obs, reward, done, truncated, info = env.step(action)
        #
        # ⚠️ 注意：
        # - 这里用的是action（active_policy选择的），不是oracle_action
        # - action决定"我们会进入什么状态"
        # - oracle_action决定"在这个状态上学什么行为"

        # ----- TODO 2.8: 处理episode结束 -----
        # if done or truncated:
        #     obs, _ = env.reset()

    # ========== TODO 3: 返回轨迹 ==========
    # return trajectory
```

### 完整实现示例

这里提供一个完整的参考实现（你可以先自己尝试，遇到困难再查看）：

<details>
<summary>点击展开完整代码</summary>

```python
def sample_trajectory(oracle, policy, env, n_steps, beta=0.5, use_criticality=True):
    """采样 n_steps 个样本（Mask PPO版本）"""

    # 1. 初始化
    trajectory = []
    if policy is None:
        policy = oracle  # 第一轮迭代，没有policy，用oracle

    obs, _ = env.reset()
    episode_steps = 0

    # 2. 采样循环
    while len(trajectory) < n_steps:
        # 2.1 混合策略 - 决定谁来执行动作
        use_oracle_for_action = np.random.binomial(1, beta) == 1
        active_policy = oracle if use_oracle_for_action else policy

        # 2.2 获取Action Mask
        mask = (obs == 0).astype(bool)
        mask_tensor = torch.tensor(mask).unsqueeze(0)

        # 2.3 使用active_policy选择动作
        if isinstance(active_policy, DecisionTreeClassifier):
            # 决策树预测
            action = active_policy.predict(obs.reshape(1, -1))[0]

            # 检查合法性
            if not mask[action]:
                # 预测了非法动作，随机选择合法动作
                valid_actions = np.where(mask)[0]
                action = np.random.choice(valid_actions)
        else:
            # MaskablePPO预测
            action, _ = active_policy.predict(
                obs,
                action_masks=mask_tensor,
                deterministic=True
            )

        # 2.4 获取Oracle的动作作为标签
        if use_oracle_for_action:
            # 优化：如果已经用oracle选择了，复用action
            oracle_action = action
        else:
            # 否则调用oracle获取标签
            oracle_action, _ = oracle.predict(
                obs,
                action_masks=mask_tensor,
                deterministic=True
            )

        # 2.5 计算criticality
        if use_criticality:
            try:
                weight = compute_criticality(oracle, obs)[0]
            except Exception as e:
                print(f"Warning: criticality计算失败: {e}")
                weight = 1.0
        else:
            weight = 1.0

        # 2.6 保存样本
        trajectory.append((obs.copy(), oracle_action, weight))

        # 2.7 执行动作
        obs, reward, done, truncated, info = env.step(action)
        episode_steps += 1

        # 2.8 处理episode结束
        if done or truncated:
            obs, _ = env.reset()
            episode_steps = 0

    # 3. 返回轨迹
    return trajectory
```

</details>

---

## 🧪 测试策略

### 第1步：简化版测试

先实现最简单的版本，验证基本逻辑：

```python
def sample_trajectory_simple(oracle, env, n_steps):
    """简化版：只用Oracle采样，不计算criticality"""
    trajectory = []
    obs, _ = env.reset()

    while len(trajectory) < n_steps:
        # 获取mask
        mask = (obs == 0).astype(bool)
        mask_tensor = torch.tensor(mask).unsqueeze(0)

        # Oracle选择动作
        action, _ = oracle.predict(obs, action_masks=mask_tensor, deterministic=True)

        # 固定权重
        weight = 1.0

        # 保存
        trajectory.append((obs.copy(), action, weight))

        # 执行
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            obs, _ = env.reset()

    return trajectory

# 测试
env = gym.make('TicTacToe-v0', opponent_type='minmax')
oracle = MaskablePPO.load('log/oracle_TicTacToe_ppo_aggressive.zip', env=env)

trajectory = sample_trajectory_simple(oracle, env, n_steps=100)
print(f"✓ 采样了 {len(trajectory)} 个样本")

# 检查数据格式
obs, action, weight = trajectory[0]
assert obs.shape == (9,), f"状态shape错误: {obs.shape}"
assert 0 <= action <= 8, f"动作超出范围: {action}"
assert obs[action] == 0, f"非法动作！obs[{action}] = {obs[action]}"
print("✓ 数据格式正确")
```

### 第2步：添加Criticality

```python
def sample_trajectory_with_criticality(oracle, env, n_steps):
    """添加criticality计算"""
    trajectory = []
    obs, _ = env.reset()

    while len(trajectory) < n_steps:
        mask = (obs == 0).astype(bool)
        mask_tensor = torch.tensor(mask).unsqueeze(0)

        action, _ = oracle.predict(obs, action_masks=mask_tensor, deterministic=True)

        # 添加criticality计算
        weight = compute_criticality(oracle, obs)[0]

        trajectory.append((obs.copy(), action, weight))

        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            obs, _ = env.reset()

    return trajectory

# 测试
trajectory = sample_trajectory_with_criticality(oracle, env, n_steps=100)

# 检查权重范围
weights = [w for _, _, w in trajectory]
print(f"权重范围: {min(weights):.3f} - {max(weights):.3f}")
print(f"平均权重: {np.mean(weights):.3f}")
```

### 第3步：完整版测试（含混合策略）

```python
def test_full_sample_trajectory():
    """测试完整的sample_trajectory"""
    env = gym.make('TicTacToe-v0', opponent_type='minmax')
    oracle = MaskablePPO.load('log/oracle_TicTacToe_ppo_aggressive.zip', env=env)

    # 先训练一个简单的Tree
    print("训练初始决策树...")
    simple_trajectory = sample_trajectory_simple(oracle, env, n_steps=1000)
    X = np.array([s[0] for s in simple_trajectory])
    y = np.array([s[1] for s in simple_trajectory])
    w = np.array([s[2] for s in simple_trajectory])

    tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree.fit(X, y, sample_weight=w)
    print("✓ 初始决策树训练完成")

    # 测试不同的beta值
    for beta in [1.0, 0.5, 0.0]:
        print(f"\n测试 beta={beta}")
        trajectory = sample_trajectory(
            oracle, tree, env,
            n_steps=100,
            beta=beta,
            use_criticality=True
        )

        print(f"  采样数量: {len(trajectory)}")

        # 检查合法性
        illegal_count = 0
        for obs, action, weight in trajectory:
            if obs[action] != 0:
                illegal_count += 1

        print(f"  非法动作: {illegal_count}")
        assert illegal_count == 0, "存在非法动作！"

        # 统计oracle_action vs tree_action的差异
        # （这可以间接验证混合策略是否工作）
        # 注意：我们只保存了oracle_action，所以这里只能统计oracle的选择分布
        actions = [a for _, a, _ in trajectory]
        print(f"  动作分布: {np.bincount(actions, minlength=9)}")

    print("\n✓ 所有测试通过！")

# 运行测试
test_full_sample_trajectory()
```

---

## ⚠️ 常见错误

### 错误1：混淆action和oracle_action

```python
# ❌ 错误：保存了执行的动作
active_policy = [tree, oracle][use_oracle]
action = active_policy.predict(...)
trajectory.append((obs, action, weight))  # 如果用tree执行，这里是tree的动作！

# ✅ 正确：保存Oracle的动作
active_policy = [tree, oracle][use_oracle]
action = active_policy.predict(...)        # 用于env.step()
oracle_action = oracle.predict(...)        # 用于训练标签
trajectory.append((obs, oracle_action, weight))
```

### 错误2：忘记Action Masking

```python
# ❌ 错误：Tree可能预测非法动作
action = tree.predict(obs.reshape(1, -1))[0]
env.step(action)  # 可能crash或-10惩罚

# ✅ 正确：检查并修正
action = tree.predict(obs.reshape(1, -1))[0]
if not mask[action]:
    valid_actions = np.where(mask)[0]
    action = np.random.choice(valid_actions)
env.step(action)
```

### 错误3：忘记copy观察

```python
# ❌ 错误：所有样本的obs指向同一对象
trajectory.append((obs, oracle_action, weight))
# 结果：所有样本的obs都是最后一个状态！

# ✅ 正确：复制
trajectory.append((obs.copy(), oracle_action, weight))
```

### 错误4：Beta理解错误

```python
# ❌ 错误理解
beta = 0.5  # 我以为这表示50%的训练数据来自Oracle

# ✅ 正确理解
beta = 0.5  # 50%的时间用Oracle执行动作（探索Oracle的轨迹）
            # 但100%的标签都来自Oracle！
```

---

## 🔄 与VIPER原始代码的对比

| 方面 | VIPER原始 (train/viper.py) | Mask PPO版 |
|------|---------------------------|-----------|
| **Oracle类型** | PPO/A2C | MaskablePPO |
| **Policy类型** | DecisionTree | DecisionTree (相同) |
| **动作选择** | `policy.predict(obs)` | `policy.predict(obs, action_masks=mask)` |
| **合法性检查** | 不需要 | **Tree预测后需要检查** |
| **环境特殊处理** | Pong需要`extract_obs` | TicTacToe直接用obs |
| **Step返回值** | 兼容4值和5值 | 直接用5值 (gym新API) |

---

## 📊 数据流图解

```
采样循环每一步的数据流：

1. 观察状态 obs
   ↓
2. 确定active_policy（根据beta）
   ├─ beta=1.0 → active_policy = oracle
   ├─ beta=0.5 → active_policy = 50% oracle, 50% tree
   └─ beta=0.0 → active_policy = tree
   ↓
3. active_policy选择action
   ├─ Tree: action = tree.predict(obs) → 检查合法性
   └─ Oracle: action = oracle.predict(obs, mask)
   ↓
4. Oracle标注oracle_action
   ├─ 如果active_policy是oracle → oracle_action = action (复用)
   └─ 如果active_policy是tree  → oracle_action = oracle.predict(obs, mask)
   ↓
5. 计算criticality
   weight = compute_criticality(oracle, obs)
   ↓
6. 保存样本
   trajectory.append((obs.copy(), oracle_action, weight))
                     ^^^^^^^^     ^^^^^^^^^^^^^  ^^^^^^
                     当前状态      Oracle的标签    重要度
   ↓
7. 执行action（不是oracle_action！）
   next_obs = env.step(action)
   ↓
8. 更新obs，继续循环
```

---

## 🎓 核心要点总结

1. **混合策略（Beta）**：控制谁来执行动作，平衡探索和利用
2. **Oracle标签**：无论谁执行，标签永远来自Oracle（模仿学习的本质）
3. **Criticality**：给重要状态更高权重，优先学好关键决策
4. **Action Masking**：TicTacToe特有，必须保证动作合法
5. **两个动作**：
   - `action`: 用于`env.step()`，决定进入什么状态
   - `oracle_action`: 用于训练，决定学什么行为

---

## 📚 参考资料

- **原始VIPER实现**: `train/viper.py` 第137-184行
- **VIPER论文**: [Verifiably Safe RL (Bastani et al. 2018)](https://arxiv.org/abs/1805.11328)
- **DAgger算法**: [Dataset Aggregation (Ross et al. 2011)](https://arxiv.org/abs/1011.0686)
- **学习指南**: [VIPER_LEARNING_GUIDE.md](VIPER_LEARNING_GUIDE.md)

---

## ✅ 实现清单

在完成`sample_trajectory`之前，确认：

- [ ] 理解了Beta参数的作用（混合策略）
- [ ] 理解了为什么标签永远来自Oracle
- [ ] 理解了action和oracle_action的区别
- [ ] 知道如何处理Action Masking
- [ ] 知道为什么要用obs.copy()

完成实现后，验证：

- [ ] 能采样指定数量的样本
- [ ] 数据格式正确：(obs, oracle_action, weight)
- [ ] 所有oracle_action都是合法的
- [ ] 不同beta值有不同的采样行为
- [ ] Weight在合理范围内

---

## 🎯 深度理解：三个"灵魂问题"

在开始实现之前，确保你能回答这三个问题：

### 问题1：为什么要用beta混合策略？直接用Oracle采样不行吗？

<details>
<summary>点击查看详细解答</summary>

**短答案**：不行！会产生**协变量偏移（Covariate Shift）**问题。

**详细解释**：

```
场景：训练井字棋决策树

如果只用Oracle采样：
┌─────────────────────────────────────────┐
│ 第1轮：Oracle采样 → 收集1000个样本      │
│        所有状态都是"Oracle会遇到的状态" │
│        训练Tree1                        │
└─────────────────────────────────────────┘
          ↓
┌─────────────────────────────────────────┐
│ 测试Tree1：                             │
│   Tree1走了几步 → 犯了错误              │
│   → 进入"糟糕状态"（Oracle不会遇到）   │
│   → Tree1没见过这种状态                 │
│   → 不知道怎么办 → 性能崩溃！          │
└─────────────────────────────────────────┘

问题根源：
  训练分布（Oracle的轨迹）≠ 测试分布（Tree的轨迹）
```

**VIPER的解决方案（DAgger）**：

```
第1轮：beta=1.0（100% Oracle采样）
  → 收集Oracle的轨迹
  → 训练Tree1

第2轮：beta=0.5（50% Tree1采样 + 50% Oracle采样）
  → Tree1会犯错，进入"糟糕状态"
  → 在这些糟糕状态上，记录Oracle会怎么做
  → 训练Tree2（学会在糟糕状态下补救）

第3轮：beta=0.25（75% Tree2采样）
  → 更多让Tree2自己走
  → 继续学习Tree2会遇到的新状态
  ...
```

**数学表达**：
```
普通模仿学习：D_train = D_oracle ≠ D_test = D_tree
DAgger：D_train = 混合(D_oracle, D_tree) ≈ D_test
```

</details>

---

### 问题2：为什么标签永远来自Oracle？不应该学习自己的动作吗？

<details>
<summary>点击查看详细解答</summary>

**短答案**：VIPER是**模仿学习**，不是**强化学习**。目标是学Oracle的行为，不是自我进化。

**详细解释**：

```
错误理解（强化学习思维）：
  Tree执行动作 → 得到奖励 → 学习"哪些动作奖励高"
  ❌ 这是RL，不是VIPER

正确理解（模仿学习思维）：
  在任何状态下 → 看Oracle会做什么 → 学习模仿Oracle
  ✅ 这是模仿学习
```

**为什么不学自己的动作？**

| 学习对象 | 问题 | 示例 |
|---------|------|------|
| **学Tree的动作** | 会强化错误 | Tree选了糟糕的动作 → 学习这个糟糕动作 → 越学越差 |
| **学Oracle的动作** ✅ | 永远学最优策略 | Tree选了糟糕的动作 → 学Oracle的补救 → 下次遇到知道正确做法 |

**井字棋示例**：

```
状态：即将输棋
┌───┬───┬───┐
│ O │ O │   │  ← 对手马上要赢（位置2）
├───┼───┼───┤
│   │ X │   │
├───┼───┼───┤
│   │   │   │
└───┴───┴───┘

Tree（菜鸟）：选择位置6（随便下）
Oracle（高手）：选择位置2（堵住对手）

如果学Tree的动作：
  dataset.append((obs, 6, weight))  # 学习"在这种状态下选6"
  → 下次遇到还是输 ❌

如果学Oracle的动作：
  dataset.append((obs, 2, weight))  # 学习"在这种状态下选2"
  → 下次遇到知道要堵 ✅
```

**关键公式**：
```python
# 模仿学习的目标
minimize Σ loss(tree(s), oracle(s))  # 让Tree接近Oracle
                      ^^^^^^^^  # 永远是Oracle的动作！

# 而不是强化学习的目标
maximize Σ reward(s, tree(s))  # 让Tree自己探索高奖励动作
```

</details>

---

### 问题3：action和oracle_action有什么区别？为什么需要两个？

<details>
<summary>点击查看详细解答</summary>

**短答案**：
- `action`: 决定**进入哪些状态**（探索）
- `oracle_action`: 决定**学什么行为**（标注）

**详细解释**：

```python
# 在sample_trajectory的每一步
active_policy = [tree, oracle][np.random.binomial(1, beta)]
action = active_policy.predict(obs)        # action: 用于执行
oracle_action = oracle.predict(obs)        # oracle_action: 用于标注

# 执行：用action
next_obs = env.step(action)  # action决定下一个状态

# 保存：用oracle_action
dataset.append((obs, oracle_action, weight))  # oracle_action是训练标签
```

**为什么不用同一个？**

| 情况 | action来源 | oracle_action来源 | 作用 |
|------|-----------|------------------|------|
| **beta=1.0, active=oracle** | oracle | oracle | 探索Oracle的轨迹 |
| **beta=0.5, active=oracle** | oracle | oracle | 探索Oracle的轨迹 |
| **beta=0.5, active=tree** | **tree** | **oracle** | 探索Tree的错误，但学Oracle的补救！⭐ |

**关键案例**：

```
第2轮，beta=0.5，某一步active_policy=tree

Tree选择：action = 6（错误的动作）
Oracle选择：oracle_action = 2（正确的动作）

执行：
  next_obs = env.step(action=6)  # 用Tree的动作执行
  → 进入一个"糟糕状态"（Tree自己犯错后的状态）

保存：
  dataset.append((obs, oracle_action=2, weight))
  → 记录"在这个状态，Oracle会选2"

结果：
  - action=6 让我们进入了Tree会遇到的糟糕状态（探索）
  - oracle_action=2 告诉Tree在这里应该怎么做（标注）
  - 下次训练，Tree学会"哦，在那个状态应该选2，不是6"
```

**如果混淆会怎样？**

```python
# ❌ 错误：用action作为标签
dataset.append((obs, action, weight))

当active_policy=tree时：
  - action是tree的选择（可能是错的）
  - 标签变成tree的动作
  → Tree学习自己的错误 → 越学越差！
```

**数据流总结**：

```
┌──────────────┐
│  当前状态 obs │
└──────┬───────┘
       │
       ├─────────────────────┐
       │                     │
       ↓                     ↓
 ┌───────────┐         ┌──────────┐
 │active_policy│       │  oracle  │
 └─────┬─────┘         └────┬─────┘
       │                    │
       ↓                    ↓
    action             oracle_action
       │                    │
       ↓                    │
  env.step()                │
  决定下一个状态             │
                            ↓
                    保存到dataset
                    作为训练标签
```

</details>

---

## 🧪 验证你是否真正理解

### 测试1：判断对错

```python
# 场景：beta=0.5, active_policy=tree

action = tree.predict(obs)        # Tree选了动作5
oracle_action = oracle.predict(obs)  # Oracle选了动作2

# 下面哪个是对的？

# A.
env.step(action=5)
dataset.append((obs, 5, weight))

# B.
env.step(action=2)
dataset.append((obs, 2, weight))

# C.
env.step(action=5)
dataset.append((obs, 2, weight))

# D.
env.step(action=2)
dataset.append((obs, 5, weight))
```

<details>
<summary>点击查看答案</summary>

**答案：C**

解释：
- `env.step(action=5)`: 用Tree的动作执行，进入Tree会遇到的状态（探索）
- `dataset.append((obs, 2, weight))`: 保存Oracle的动作作为标签（学习）

- **A错误**：保存了Tree的动作，会学习错误
- **B错误**：用Oracle执行，无法探索Tree的错误状态
- **D错误**：完全反了

</details>

---

### 测试2：预测结果

```python
# 代码
trajectory = []
for i in range(4):
    obs = np.array([i, i, i])
    trajectory.append((obs, i, 1.0))

# trajectory[0][0] 的值是多少？
# A. [0, 0, 0]
# B. [3, 3, 3]
# C. [1, 1, 1]
# D. 报错
```

<details>
<summary>点击查看答案</summary>

**答案：B - [3, 3, 3]**（不是 [0, 0, 0]！）

原因：
```python
obs = np.array([i, i, i])  # 创建数组
trajectory.append((obs, i, 1.0))  # 保存引用，不是值！

# 问题：所有元素都保存了同一个obs对象的引用
# 循环结束后，obs = [3, 3, 3]
# trajectory[0][0] 指向这个对象 → [3, 3, 3]
```

**正确写法**：
```python
trajectory.append((obs.copy(), i, 1.0))  # 复制！
```

这就是为什么文档中一直强调要用 `obs.copy()`！

</details>

---

### 测试3：Beta的作用

```python
# 问题：以下哪个说法是对的？

# A. beta=1.0 表示100%的标签来自Oracle
# B. beta=1.0 表示100%的动作执行来自Oracle
# C. beta=0.5 表示50%的标签来自Oracle
# D. beta越大，Tree训练得越快
```

<details>
<summary>点击查看答案</summary>

**答案：B**

解释：
- **A错误**：标签**永远100%**来自Oracle，与beta无关
- **B正确**：beta控制执行动作的策略选择
- **C错误**：标签**永远100%**来自Oracle，不是50%
- **D错误**：beta控制采样策略，不直接影响训练速度

**关键理解**：
```python
# beta只影响这一行
active_policy = [tree, oracle][np.random.binomial(1, beta)]

# 这一行永远不变（标签永远来自Oracle）
oracle_action = oracle.predict(...)
dataset.append((obs, oracle_action, weight))
```

</details>

---

## 🔍 关键代码片段深度解析

### 片段1：混合策略（train/viper.py 第42行）

```python
active_policy = [policy, oracle][np.random.binomial(1, beta)]
```

**逐步拆解**：

```python
# 第1步：理解 np.random.binomial(1, beta)
coin_flip = np.random.binomial(1, beta)
# 这是一个"有偏硬币"
# - 返回1的概率 = beta
# - 返回0的概率 = 1-beta
#
# beta=1.0 → 100%返回1
# beta=0.5 → 50%返回1, 50%返回0
# beta=0.0 → 100%返回0

# 第2步：理解列表索引
policies = [policy, oracle]
#           ^^^^^^  ^^^^^^
#           索引0    索引1
#
# 如果coin_flip=0 → 选择policy
# 如果coin_flip=1 → 选择oracle

# 第3步：组合起来
active_policy = [policy, oracle][np.random.binomial(1, beta)]
#               ^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^
#               候选策略列表      随机选择索引（0或1）
```

**等价写法**（更清晰）：

```python
# 写法1：
if np.random.binomial(1, beta) == 1:
    active_policy = oracle
else:
    active_policy = policy

# 写法2：
if np.random.random() < beta:  # random()返回[0,1)的均匀分布
    active_policy = oracle
else:
    active_policy = policy

# 写法3（推荐用于你的实现）：
use_oracle_for_action = np.random.binomial(1, beta) == 1
active_policy = oracle if use_oracle_for_action else policy
```

---

### 片段2：获取Oracle标签（train/viper.py 第53-58行）

```python
if not isinstance(active_policy, DecisionTreeClassifier):
    oracle_action = action
else:
    oracle_action = oracle.predict(obs, deterministic=True)[0]
```

**为什么这样判断？**

```python
# 情况1：active_policy 是 oracle
if not isinstance(active_policy, DecisionTreeClassifier):  # True
    oracle_action = action  # action已经是oracle.predict()的结果，直接复用
    # 优化：避免重复调用oracle.predict()

# 情况2：active_policy 是 tree
else:  # isinstance(active_policy, DecisionTreeClassifier) == True
    oracle_action = oracle.predict(obs, deterministic=True)[0]
    # 必须调用：因为action是tree.predict()的结果，需要oracle的标签
```

**Mask PPO版本的改进**：

```python
# 原始VIPER的优化在Mask PPO中不太适用，因为需要mask_tensor
# 更清晰的写法：

if use_oracle_for_action:
    oracle_action = action  # 复用
else:
    # 重新调用（需要mask）
    oracle_action, _ = oracle.predict(
        obs,
        action_masks=mask_tensor,
        deterministic=True
    )
```

---

### 片段3：保存样本（train/viper.py 第73行）

```python
trajectory += list(zip(obs, oracle_action, state_loss))
```

**等价写法**（TicTacToe单步版本）：

```python
trajectory.append((obs.copy(), oracle_action, state_loss))
```

**为什么原始代码用`+=`和`zip`？**

```python
# 原始VIPER处理向量化环境（多个环境并行）
# obs.shape = (n_envs, obs_dim)
# oracle_action.shape = (n_envs,)
# state_loss.shape = (n_envs,)

# zip会生成多个样本
list(zip(obs, oracle_action, state_loss))
# 结果：[(obs[0], action[0], loss[0]),
#       (obs[1], action[1], loss[1]),
#       ...]

# TicTacToe是单环境，直接append即可
trajectory.append((obs.copy(), oracle_action, state_loss))
```

---

## 📚 算法对比：VIPER在模仿学习家族中的位置

| 算法 | 数据来源 | 标签来源 | 特点 | 适用场景 |
|------|---------|---------|------|---------|
| **行为克隆 (BC)** | Oracle采样 | Oracle标注 | 最简单 | 有大量专家数据 |
| **DAgger** | **混合采样** | Oracle标注 | 解决协变量偏移 | 有交互式Oracle |
| **VIPER** | **混合采样** | Oracle标注 + **Criticality加权** | 提取可解释决策树 | 需要可解释性 |
| **强化学习 (RL)** | 自我探索 | **环境奖励** | 无需专家 | 无专家，从零学习 |

**VIPER = DAgger + 决策树 + Criticality加权**

---

## 🌳 扩展：使用回归树（输出动作得分）

### 背景

VIPER原始实现使用**分类树**（DecisionTreeClassifier），直接预测一个动作：

```python
# 分类树
trajectory = [(obs, action, weight), ...]  # action是0-8的整数
tree = DecisionTreeClassifier()
tree.fit(X, y)
action = tree.predict(obs)  # 输出单个动作
```

但对于board game，你可能想使用**回归树**（DecisionTreeRegressor），输出每个动作的得分：

```python
# 回归树
trajectory = [(obs, action_scores, weight), ...]  # action_scores是(9,)数组
tree = DecisionTreeRegressor()
tree.fit(X, y)  # y.shape = (N, 9)
scores = tree.predict(obs)  # 输出9个得分
action = argmax(scores * mask)  # 结合mask选最高分
```

---

### 两种方法对比

| 方面 | 分类树 | 回归树 |
|------|--------|--------|
| **训练标签** | 单个动作 (int) | 所有动作的得分 (9维向量) |
| **标签维度** | (N,) | (N, 9) |
| **Tree类型** | DecisionTreeClassifier | DecisionTreeRegressor |
| **预测输出** | 单个动作 | 9个得分 |
| **推理时** | 直接输出 → 需检查mask | 选最高分的合法动作 |
| **信息量** | 只知道最优动作 | 知道所有动作的相对价值 ✅ |
| **可解释性** | "在这里选动作4" | "在这里，动作4得分0.8，动作2得分0.3..." ✅ |
| **Tree大小** | 较小 | 较大 |

**推荐回归树的理由**：
- ✅ 更接近PPO的本质（PPO输出每个动作的logits）
- ✅ 保留了动作之间的相对优劣信息
- ✅ 结合Action Masking更自然
- ✅ 可解释性更强（能看到"为什么不选其他动作"）

---

### PPO输出的是什么？

#### 关键理解：PPO ≠ Q-Learning

```python
# ❌ Q-Learning (DQN) - 值函数方法
# 输出：Q(s,a) - 每个动作的期望回报
Q_values = model(obs)  # [Q(s,0), Q(s,1), ..., Q(s,8)]
action = argmax(Q_values)

# ✅ PPO - 策略梯度方法
# 输出：π(a|s)的logits - 动作概率的未归一化值
obs → MLP → features → action_net → logits → softmax → probabilities
                                     ^^^^^^
                                   我们要这个！
```

#### PPO的输出层

```python
# PPO策略网络
logits = policy.action_net(latent_features)  # 未归一化得分
probs = softmax(logits)  # 归一化后得到概率分布
action = sample(probs)  # 或 argmax(probs) if deterministic
```

**示例**：

```python
obs = np.array([0, -1, 1, 0, 0, 0, 0, 0, 0])

# PPO输出logits
logits = array([2.1, -5.0, -3.2, 1.8, 0.5, -1.2, 0.8, 1.5, -0.5])
#               ^^^   ^^^^        ^^^
#              最高   很低        次高

# Softmax后得到概率
probs = softmax(logits)
# array([0.45, 0.0, 0.0, 0.32, 0.05, 0.01, 0.08, 0.09, 0.0])
#        ^^^^                ^^^^
#       45%概率             32%概率

# PPO选择动作
action = argmax(logits)  # deterministic=True → 选动作0
# 或
action = np.random.choice(9, p=probs)  # deterministic=False → 按概率采样
```

---

### 回归树应该学什么？Logits vs Log_probs

#### 三个选项

```python
# 选项1：Logits（未归一化）
logits = distribution.distribution.logits  # [-∞, +∞]

# 选项2：Log Probabilities
log_probs = distribution.distribution.log_prob(actions)  # (-∞, 0]

# 选项3：Probabilities
probs = softmax(logits)  # [0, 1], sum=1
```

#### 为什么用Logits而不是Log_probs？

<details>
<summary>点击查看详细解答</summary>

**1. 定义区别**

```python
# Logits: 策略网络的原始输出（未归一化）
logits = policy.action_net(features)
# 例如：[2.1, -5.0, 1.8, 0.5, ...]
# 特点：
# - 没有概率意义
# - 范围：(-∞, +∞)
# - 相对大小决定概率

# Log Probabilities: 归一化后取log
log_probs = log(softmax(logits))
# 例如：[-0.8, -12.3, -1.1, -2.9, ...]
# 特点：
# - 有概率意义：exp(log_prob) = prob
# - 范围：(-∞, 0]
# - 最大值接近0，其他都是负数
```

**2. 数学关系**

```python
# 从logits到log_probs的转换
log_probs = logits - log_sum_exp(logits)
# 或
log_probs = log_softmax(logits)

# 示例
logits = [2.1, -5.0, 1.8]
log_sum_exp(logits) = log(e^2.1 + e^-5.0 + e^1.8) = 2.9

log_probs = [2.1 - 2.9, -5.0 - 2.9, 1.8 - 2.9]
          = [-0.8, -7.9, -1.1]
```

**3. 为什么回归树更适合学Logits？**

| 方面 | Logits | Log_probs |
|------|--------|-----------|
| **数值范围** | (-∞, +∞) | (-∞, 0] |
| **分散度** | 大（易区分） | 小（挤在负数区间） |
| **是否需要归一化** | 否 | 是（依赖于其他动作） |
| **回归树学习难度** | 容易 ✅ | 困难 |

**具体示例**：

```python
# 场景：3个动作，动作0最好，动作1次好，动作2最差

# Logits
logits = [2.1, 1.8, -5.0]
# 差异：2.1 - 1.8 = 0.3,  1.8 - (-5.0) = 6.8
# 回归树容易学：直接比较大小

# Log_probs
log_probs = [-0.8, -1.1, -7.9]
# 差异：-0.8 - (-1.1) = 0.3,  -1.1 - (-7.9) = 6.8
# 看起来差异相同？

# 但问题来了：如果场景变化
# 场景2：4个动作
logits_2 = [2.1, 1.8, -5.0, 0.5]
log_probs_2 = log_softmax(logits_2)
          = [-1.3, -1.6, -8.4, -2.9]
# 注意：同样的前3个动作，log_probs变了！
# 这是因为log_probs依赖于所有动作的归一化

# 但logits不变！
# [2.1, 1.8, -5.0] 无论后面有没有第4个动作
```

**4. 归一化的问题**

```python
# Log_probs的问题：耦合性
# 动作A的log_prob不仅取决于A的质量，还取决于其他动作

# 例子：井字棋
# 状态1：9个合法动作
logits_1 = [2.1, 1.8, 1.5, 1.2, 0.8, 0.5, 0.2, -0.1, -0.5]
log_probs_1 = log_softmax(logits_1)
# 动作0的log_prob ≈ -0.5

# 状态2：只有2个合法动作（但动作0的质量相同）
logits_2 = [2.1, 1.8, -100, -100, -100, ...]  # mask后
log_probs_2 = log_softmax(logits_2)
# 动作0的log_prob ≈ -0.3  <- 变了！

# 这导致回归树很难学：
# 同样的状态特征 → 但标签（log_prob）不一致
```

**5. 回归树的视角**

```python
# 回归树的目标：学习 f(state) → scores

# 用Logits训练
X = [state_1, state_2, state_3]
y_logits = [
    [2.1, 1.8, -5.0, ...],  # 直接的"好坏"评分
    [1.5, 2.3, 0.8, ...],
    ...
]
tree.fit(X, y_logits)
# 树学到："在state_1，动作0得分2.1，动作1得分1.8"
# 清晰！每个动作有独立的得分

# 用Log_probs训练
y_log_probs = [
    [-0.8, -1.1, -7.9, ...],  # 受归一化影响的值
    [-1.2, -0.7, -2.5, ...],
    ...
]
tree.fit(X, y_log_probs)
# 树学到："在state_1，动作0得分-0.8，动作1得分-1.1"
# 但这个-0.8依赖于其他动作！树很难理解这种依赖关系
```

**6. 实验证据**

```python
# 一个简化的实验

# 方法1：用logits
tree_logits = DecisionTreeRegressor()
tree_logits.fit(X, logits)
scores = tree_logits.predict(test_state)  # [2.0, 1.7, -4.8, ...]
action = argmax(scores[mask])  # 工作良好 ✅

# 方法2：用log_probs
tree_log_probs = DecisionTreeRegressor()
tree_log_probs.fit(X, log_probs)
scores = tree_log_probs.predict(test_state)  # [-0.9, -1.2, -8.1, ...]
action = argmax(scores[mask])  # 也能工作，但...

# 问题：
# - log_probs的数值范围小，树不容易分裂
# - log_probs的依赖性导致泛化困难
# - 实际测试中，logits训练的树性能更好
```

**7. 总结**

| | Logits | Log_probs |
|---|--------|-----------|
| **来源** | 策略网络直接输出 | Logits + log_softmax |
| **独立性** | 每个动作独立 ✅ | 耦合（归一化） ❌ |
| **数值范围** | 大（好区分） ✅ | 小（挤在负数） ❌ |
| **物理意义** | "未归一化得分" | "对数概率" |
| **回归树适配** | 优秀 ✅ | 可行但次优 |

**结论**：回归树应该学习**logits**！

</details>

---

### 实现：从MaskablePPO获取Logits

```python
def get_oracle_action_logits(oracle, obs):
    """从MaskablePPO获取每个动作的logits

    PPO输出的是策略π(a|s)的logits（未归一化的得分）
    这是策略网络action_net的原始输出

    Args:
        oracle: MaskablePPO模型
        obs: 观察 (9,)

    Returns:
        logits: np.array of shape (9,), 每个动作的logit值
    """
    with torch.no_grad():
        # 1. 准备输入
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        obs_tensor = obs_tensor.to(oracle.device)

        # 2. 方法1：通过distribution（推荐，更简洁）
        distribution = oracle.policy.get_distribution(obs_tensor)
        logits = distribution.distribution.logits.cpu().numpy()[0]  # (9,)

        # 方法2：直接通过网络（等价，更底层）
        # features = oracle.policy.extract_features(obs_tensor)
        # latent_pi, _ = oracle.policy.mlp_extractor(features)
        # logits = oracle.policy.action_net(latent_pi).cpu().numpy()[0]

        return logits
```

**注意**：
- 这是获取**logits**，不是log_probs
- Logits是策略网络的**原始输出**，没有经过softmax
- 范围是 `(-∞, +∞)`，不是概率
- ⚠️ **不要mask！** 获取所有9个动作的原始logits，不要将非法动作设为-inf

---

## ❓ 常见疑问：Mask和归一化

### 疑问1：应该学习mask之前还是之后的logits？

**答案：学习mask之前的原始logits ✅**

<details>
<summary>点击查看详细解释</summary>

**错误做法❌**：
```python
# 获取原始logits
logits = oracle.policy.action_net(features)  # [2.1, -5.0, 1.8, 0.5, -1.2, ...]

# mask非法动作
mask = (obs == 0).astype(bool)  # [True, False, True, ...]
masked_logits = logits.clone()
masked_logits[~mask] = -np.inf  # [2.1, -inf, 1.8, 0.5, -inf, ...]

# 保存masked logits（错误！）
trajectory.append((obs, masked_logits, weight))
```

**正确做法✅**：
```python
# 获取原始logits
logits = oracle.policy.action_net(features)  # [2.1, -5.0, 1.8, 0.5, -1.2, ...]

# 直接保存原始logits（正确！）
trajectory.append((obs, logits, weight))
```

**为什么？**

1. **状态独立性**
```python
# 场景1：obs = [0, 1, 0, 0, 0, ...]  (8个合法动作)
原始logits:  [2.1, -5.0, 1.8, 0.5, -1.2, ...]
Masked后:    [2.1, -inf, 1.8, 0.5, -inf, ...]

# 场景2：obs = [0, 0, 0, 0, 0, ...]  (9个合法动作)
原始logits:  [2.1, -5.0, 1.8, 0.5, -1.2, ...]
Masked后:    [2.1, -5.0, 1.8, 0.5, -1.2, ...]  <- 同样的obs特征，不同的输出！

# 问题：mask依赖于当前状态，不同状态的mask不同
# 回归树很难学：相同的棋盘特征 → 但logits不一致
```

2. **决策树的学习目标**
```python
# 回归树要学：f(棋盘状态) → logits

# 用原始logits：
f([0, 1, 0, ...]) → [2.1, -5.0, 1.8, ...]
# ✅ 树学到："位置0得分2.1，位置1（已占）得分-5.0，位置2得分1.8"
# 推理时再mask：action = argmax(logits * mask)

# 用masked logits：
f([0, 1, 0, ...]) → [2.1, -inf, 1.8, ...]
# ❌ 树学到的是"位置1是-inf"
# 但位置1的-inf是因为当前状态占了，不是位置1本身不好
# 新状态下位置1空了，树还是输出-inf → 错误！
```

3. **泛化能力**
```python
# 训练时见过的状态
obs_train = [0, 1, 0, 0, 0, ...]
logits_train = [2.1, -inf, 1.8, ...]  # mask后

# 测试时新状态（位置1空了）
obs_test = [0, 0, 0, 0, 0, ...]
tree.predict(obs_test)
# ❌ 树可能仍然输出位置1为-inf（因为训练时学到的）

# 但如果训练时用原始logits：
logits_train = [2.1, -5.0, 1.8, ...]  # 原始
tree.predict(obs_test)  # [2.1, -5.0, 1.8, ...]
# ✅ 推理时根据mask决定是否可选：action = argmax(scores[mask])
```

**类比理解**：

想象你在教一个学生下棋：

- **错误教法**：告诉他"在这个局面，位置1不能下（因为有棋子了），得分是负无穷"
  - 学生记住："位置1总是不能下"
  - 新局面位置1空了，学生还是不敢下 ❌

- **正确教法**：告诉他"位置1本身的价值是-5.0（不太好的位置）"
  - 学生记住："位置1价值-5.0"
  - 新局面位置1空了且合法，学生会根据其他位置的价值比较 ✅

</details>

---

### 疑问2：是否需要对logits归一化？

**答案：不需要，学习原始logits最好 ✅**

<details>
<summary>点击查看详细解释</summary>

**三种选择对比**：

```python
# 选项1：原始logits（推荐）
logits = [2.1, -5.0, 1.8, 0.5, -1.2, 0.8, 1.5, -0.5, 0.2]
trajectory.append((obs, logits, weight))

# 选项2：Mask后归一化（不推荐）
mask = [True, False, True, True, False, True, True, True, True]
masked_logits = [2.1, -inf, 1.8, 0.5, -inf, 0.8, 1.5, -0.5, 0.2]
normalized = (masked_logits - mean) / std  # 只对合法动作
trajectory.append((obs, normalized, weight))

# 选项3：Softmax归一化（不推荐）
probs = softmax(logits[mask])
trajectory.append((obs, probs, weight))
```

**为什么不需要归一化？**

1. **决策树对尺度不敏感**
```python
# 决策树的分裂规则基于相对大小，不是绝对值
# 例如：if feature_0 > 1.5: action_0, else: action_1

# 原始logits: [2.1, 1.8, 0.5]
# 树学到：if ... : predict [2.1, 1.8, 0.5]

# 归一化后: [0.8, 0.6, -0.2]  (假设)
# 树学到：if ... : predict [0.8, 0.6, -0.2]

# 推理时：
action = argmax(tree.predict(obs))
# 两者结果一样！因为argmax只看相对大小
```

2. **原始logits保留了语义**
```python
# 原始logits的含义
logits = [2.1, -5.0, 1.8]
# 2.1: 好位置
# -5.0: 非常差的位置（可能是已占或战略意义低）
# 1.8: 还不错的位置

# 归一化后的含义
normalized = [1.2, -2.1, 0.9]  # 假设
# 语义变了！现在只能说相对好坏，不能说绝对好坏
```

3. **归一化增加复杂度**
```python
# 不同状态的归一化参数不同
# 状态1：9个合法动作
mean_1 = mean(logits_1)
std_1 = std(logits_1)

# 状态2：2个合法动作
mean_2 = mean(logits_2)
std_2 = std(logits_2)

# 回归树很难学：同样的位置，归一化后数值不同
```

**实验对比**：

```python
# 实验：训练两个树

# Tree A：原始logits
tree_A = DecisionTreeRegressor()
tree_A.fit(X, logits_original)

# Tree B：归一化logits
logits_normalized = (logits_original - logits_original.mean(axis=1, keepdims=True)) / logits_original.std(axis=1, keepdims=True)
tree_B = DecisionTreeRegressor()
tree_B.fit(X, logits_normalized)

# 测试
test_obs = [0, 0, 1, 0, ...]
scores_A = tree_A.predict(test_obs)
scores_B = tree_B.predict(test_obs)

action_A = argmax(scores_A[mask])
action_B = argmax(scores_B[mask])

# 结果：action_A == action_B（大部分情况）
# 但Tree A的logits更有语义，易于调试
```

**总结**：

| | 原始logits | Mask后的logits | 归一化logits |
|---|-----------|---------------|-------------|
| **状态独立性** | ✅ 独立 | ❌ 依赖mask | ❌ 依赖mask |
| **语义清晰** | ✅ 有绝对意义 | ⚠️ -inf无语义 | ❌ 相对意义 |
| **泛化能力** | ✅ 好 | ❌ 差 | ⚠️ 一般 |
| **实现复杂度** | ✅ 简单 | ⚠️ 中等 | ❌ 复杂 |
| **推荐度** | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐ |

**结论**：直接学习原始logits！

</details>

---

### 实现建议

```python
def get_oracle_action_logits(oracle, obs):
    """获取原始logits（不要mask！）"""
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        obs_tensor = obs_tensor.to(oracle.device)

        distribution = oracle.policy.get_distribution(obs_tensor)
        logits = distribution.distribution.logits.cpu().numpy()[0]

        # ✅ 直接返回原始logits
        return logits

        # ❌ 不要这样做：
        # mask = (obs == 0).astype(bool)
        # logits[~mask] = -np.inf
        # return logits

def sample_trajectory_regression(oracle, policy, env, n_steps, beta=0.5):
    """采样时保存原始logits"""
    trajectory = []
    obs, _ = env.reset()

    while len(trajectory) < n_steps:
        # ... 选择动作的逻辑 ...

        # ✅ 获取并保存原始logits
        oracle_logits = get_oracle_action_logits(oracle, obs)
        trajectory.append((obs.copy(), oracle_logits.copy(), weight))

        # ❌ 不要这样做：
        # mask = (obs == 0).astype(bool)
        # masked_logits = oracle_logits.copy()
        # masked_logits[~mask] = -np.inf
        # trajectory.append((obs.copy(), masked_logits, weight))

    return trajectory

class RegressionTreePolicy:
    """推理时才应用mask"""

    def predict(self, observation, deterministic=True):
        # 1. 预测原始logits
        logits = self.tree.predict(observation.reshape(1, -1))[0]

        # 2. 推理时才应用mask
        mask = (observation == 0).astype(bool)
        legal_actions = np.where(mask)[0]

        # 3. 选择最高logit的合法动作
        legal_logits = logits[legal_actions]
        best_idx = np.argmax(legal_logits)
        action = legal_actions[best_idx]

        return action, None
```

---

### 核心原则总结

1. **训练时**：保存原始logits（所有9个动作）
2. **推理时**：应用mask选择合法动作
3. **不要归一化**：决策树对尺度不敏感，原始logits语义更清晰

**关键理念**：
```
学习"每个位置的内在价值" ✅
而不是
学习"在当前状态哪些位置可选" ❌
```

---

### 修改后的sample_trajectory（回归树版）

```python
def sample_trajectory_regression(oracle, policy, env, n_steps, beta=0.5, use_criticality=True):
    """采样轨迹 - 回归树版本

    Returns:
        trajectory: List of (obs, logits, weight)
    """
    trajectory = []
    policy = policy or oracle

    obs, _ = env.reset()

    while len(trajectory) < n_steps:
        # 1. 混合策略
        use_oracle_for_action = np.random.binomial(1, beta) == 1
        active_policy = oracle if use_oracle_for_action else policy

        # 2. Mask
        mask = (obs == 0).astype(bool)
        mask_tensor = torch.tensor(mask).unsqueeze(0)

        # 3. 选择执行的动作
        if isinstance(active_policy, DecisionTreeRegressor):
            # 回归树：预测logits
            logits = active_policy.predict(obs.reshape(1, -1))[0]  # (9,)

            # 选择最高logit的合法动作
            masked_logits = np.full(9, -np.inf)
            masked_logits[mask] = logits[mask]
            action = np.argmax(masked_logits)
        else:  # MaskablePPO
            action, _ = active_policy.predict(
                obs,
                action_masks=mask_tensor,
                deterministic=True
            )

        # 4. 获取Oracle的logits作为标签（关键！）
        oracle_logits = get_oracle_action_logits(oracle, obs)

        # 5. 计算criticality
        if use_criticality:
            try:
                weight = compute_criticality(oracle, obs)[0]
            except:
                weight = 1.0
        else:
            weight = 1.0

        # 6. 保存（obs, oracle_logits, weight）
        # 注意：保存的是oracle_logits，不是action！
        trajectory.append((obs.copy(), oracle_logits.copy(), weight))

        # 7. 执行动作
        obs, reward, done, truncated, info = env.step(action)

        # 8. 处理episode结束
        if done or truncated:
            obs, _ = env.reset()

    return trajectory
```

---

### 训练回归树

```python
from sklearn.tree import DecisionTreeRegressor

def train_regression_tree(trajectory, max_depth=10, max_leaves=50):
    """训练回归决策树

    Args:
        trajectory: List of (obs, logits, weight)

    Returns:
        tree: DecisionTreeRegressor
    """
    # 准备数据
    X = np.array([obs for obs, _, _ in trajectory])       # (N, 9)
    y = np.array([logits for _, logits, _ in trajectory]) # (N, 9)
    weights = np.array([w for _, _, w in trajectory])     # (N,)

    print(f"训练数据: X.shape={X.shape}, y.shape={y.shape}")

    # 训练回归树
    tree = DecisionTreeRegressor(
        max_depth=max_depth,
        max_leaf_nodes=max_leaves,
        random_state=42,
        min_samples_split=10,  # 防止过拟合
        min_samples_leaf=5
    )

    tree.fit(X, y, sample_weight=weights)

    print(f"✓ 训练完成")
    print(f"  树深度: {tree.tree_.max_depth}")
    print(f"  叶子节点数: {tree.tree_.n_leaves}")

    return tree
```

---

### 推理时的Policy包装器

```python
class RegressionTreePolicy:
    """回归树策略包装器"""

    def __init__(self, tree):
        self.tree = tree
        self.n_actions = 9

    def predict(self, observation, deterministic=True):
        """预测动作（结合masking）

        Returns:
            action: 最优的合法动作
        """
        # 处理输入
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
            single_obs = True
        else:
            single_obs = False

        actions = []
        for obs in observation:
            # 预测所有动作的logits
            logits = self.tree.predict(obs.reshape(1, -1))[0]  # (9,)

            # 获取合法动作
            mask = (obs == 0).astype(bool)
            legal_actions = np.where(mask)[0]

            if len(legal_actions) == 0:
                # 无合法动作（不应发生）
                actions.append(0)
                continue

            # 选择logit最高的合法动作
            legal_logits = logits[legal_actions]
            best_idx = np.argmax(legal_logits)
            action = legal_actions[best_idx]

            actions.append(action)

        actions = np.array(actions)

        if single_obs:
            return actions[0], None
        else:
            return actions, None

    def get_action_logits(self, observation):
        """获取所有动作的logits（用于分析）

        Returns:
            logits: shape (9,)
        """
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)

        logits = self.tree.predict(observation)[0]
        return logits
```

---

### 完整测试示例

```python
def test_regression_tree_viper():
    """测试回归树版本的VIPER"""
    import gymnasium as gym
    from sb3_contrib import MaskablePPO
    import torch
    import numpy as np

    # 1. 加载Oracle
    env = gym.make('TicTacToe-v0', opponent_type='minmax')
    oracle = MaskablePPO.load('log/oracle_TicTacToe_ppo_aggressive.zip', env=env)

    # 2. 采样轨迹
    print("采样轨迹...")
    trajectory = sample_trajectory_regression(
        oracle, None, env,
        n_steps=1000,
        beta=1.0,
        use_criticality=True
    )

    # 3. 检查数据
    obs, logits, weight = trajectory[0]
    print(f"\n样本格式:")
    print(f"  obs shape: {obs.shape}")      # (9,)
    print(f"  logits shape: {logits.shape}")  # (9,)
    print(f"  weight: {weight:.3f}")
    print(f"  logits: {logits}")
    print(f"  logits范围: [{logits.min():.2f}, {logits.max():.2f}]")

    # 4. 训练回归树
    print("\n训练回归树...")
    tree = train_regression_tree(trajectory, max_depth=10, max_leaves=50)

    # 5. 包装成策略
    policy = RegressionTreePolicy(tree)

    # 6. 测试推理
    print("\n测试推理...")
    test_obs = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)
    action, _ = policy.predict(test_obs)
    logits = policy.get_action_logits(test_obs)

    print(f"测试状态: {test_obs}")
    print(f"预测logits: {logits}")
    print(f"预测动作: {action}")
    print(f"合法动作: {np.where(test_obs == 0)[0]}")

    # 7. 对比Oracle
    oracle_logits = get_oracle_action_logits(oracle, test_obs)
    oracle_action, _ = oracle.predict(test_obs, action_masks=(test_obs==0), deterministic=True)

    print(f"\nOracle对比:")
    print(f"  Oracle logits: {oracle_logits}")
    print(f"  Oracle 动作: {oracle_action}")
    print(f"  Tree vs Oracle:")
    print(f"    动作一致: {action == oracle_action}")
    print(f"    Logits相关性: {np.corrcoef(logits, oracle_logits)[0,1]:.3f}")

# 运行测试
if __name__ == '__main__':
    test_regression_tree_viper()
```

---

### ✅ 实现清单（回归树版本）

- [ ] 实现 `get_oracle_action_logits()` 函数
- [ ] 修改 `sample_trajectory` 保存 `(obs, logits, weight)`
- [ ] 确认保存的是**logits**而不是log_probs
- [ ] 使用 `DecisionTreeRegressor` 训练
- [ ] 实现 `RegressionTreePolicy` 包装器
- [ ] 测试推理时的masking逻辑
- [ ] 对比回归树和分类树的性能

---

祝你实现顺利！🚀
