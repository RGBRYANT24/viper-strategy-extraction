# TicTacToe Action Masking 训练方案 - 完整指南

> 目标：使用 Action Mask 防止非法移动，训练TicTacToe最优策略，并用VIPER提取决策树

## 目录
- [问题背景](#问题背景)
- [方案选择](#方案选择)
- [当前进度](#当前进度)
- [继续工作指南](#继续工作指南)
- [技术细节](#技术细节)
- [FAQ](#faq)

---

## 问题背景

### 原始问题
- 使用 `-10` 惩罚非法动作 → Q值污染
- 需要改用 Action Mask 在神经网络层面屏蔽非法动作

### 尝试过的方案
1. **MaskedDQNPolicy**（已实现但有问题）
   - 文件：[gym_env/masked_dqn_policy.py](gym_env/masked_dqn_policy.py)
   - 问题：DQN 的 ε-greedy exploration 绕过了 mask
   - 结果：训练初期仍有 59.4% 的非法移动

### 核心问题诊断
```python
# DQN 的 exploration 机制
if random() < exploration_rate:  # 初期 exploration_rate=1.0
    action = env.action_space.sample()  # ❌ 直接随机，绕过 policy!
else:
    action = policy.predict(obs)         # ✅ 经过 mask
```

**症状**：训练日志显示 `ep_rew_mean = -6.18`（大量 -10 惩罚）

---

## 方案选择

### 方案对比

| 方案 | 优点 | 缺点 | 工作量 | VIPER兼容 |
|------|------|------|--------|----------|
| **A. MaskablePPO** ⭐ | 原生支持mask<br>无ε-greedy问题<br>已验证兼容 | 需要sb3-contrib | ⭐⭐ | ✅ |
| B. 自定义MaskedDQN | 只需sb3<br>训练快 | 需重写collect_rollouts<br>复杂易错 | ⭐⭐⭐⭐ | ✅ |

### 推荐方案：MaskablePPO

**理由**：
1. ✅ 已验证：`MaskablePPO.predict()` 接口与 VIPER 完全兼容
2. ✅ 原生支持：专门为 action masking 设计
3. ✅ 已有代码：`train/train_delta_selfplay_ppo.py` 可直接使用
4. ✅ 无绕过问题：PPO 用策略梯度，不用随机 exploration

**验证结果**（已在服务器确认）：
```python
# MaskablePPO.predict 签名
def predict(
    observation,
    state=None,
    episode_start=None,
    deterministic=False,
    action_masks=None  # ⭐ 可选参数，VIPER调用时不需要
) -> Tuple[np.ndarray, Optional[Tuple]]

# 测试通过
action, state = model.predict(obs, deterministic=True)  # ✓ 不需要额外参数
```

---

## 当前进度

### ✅ 已完成

1. **MaskedDQNPolicy 实现**
   - 文件：[gym_env/masked_dqn_policy.py](gym_env/masked_dqn_policy.py)
   - 状态：完成但有 exploration 问题

2. **训练脚本修改**
   - 文件：[train/train_delta_selfplay.py](train/train_delta_selfplay.py)
   - 状态：使用 MaskedDQNPolicy，但训练时仍有非法移动

3. **评估工具增强**
   - 文件：[evaluate_nn_quality.py](evaluate_nn_quality.py)
   - 状态：添加了 Action Masking 测试功能

4. **PPO 训练脚本**
   - 文件：[train/train_delta_selfplay_ppo.py](train/train_delta_selfplay_ppo.py)
   - 状态：已存在且可用

5. **依赖更新**
   - 文件：[requirements.txt](requirements.txt)
   - 状态：已添加 `sb3-contrib==2.4.0`

6. **兼容性验证**
   - 状态：✅ 已确认 MaskablePPO 与 VIPER 完全兼容

### 🔲 待完成

1. 修改 [train/oracle.py](train/oracle.py) 添加 MaskablePPO 配置
2. 创建测试脚本 `test_maskable_ppo_viper.py`
3. 运行完整训练流程并验证
4. VIPER 决策树提取

---

## 继续工作指南

### 给 Claude Code 的 Prompt

```
我在使用 VIPER 框架训练 TicTacToe，需要用 Action Mask 防止非法移动。

背景：
- 尝试了 MaskedDQNPolicy，但 DQN 的 ε-greedy 会绕过 mask
- 已验证 MaskablePPO 与 VIPER 完全兼容
- 已有 train/train_delta_selfplay_ppo.py 训练脚本
- 已添加 sb3-contrib==2.4.0 依赖

现在需要：
1. 修改 train/oracle.py，添加 TicTacToe-MaskablePPO 配置（用于VIPER集成）
2. 创建测试脚本验证 MaskablePPO + VIPER 流程
3. 更新文档说明使用方法

参考：MASKED_TRAINING_GUIDE.md
```

### 任务清单

#### 任务1：修改 oracle.py

**文件**：[train/oracle.py](train/oracle.py)

**位置1**：在第1-6行导入 MaskablePPO

```python
from stable_baselines3 import DQN, PPO
try:
    from sb3_contrib import MaskablePPO
    HAS_MASKABLE_PPO = True
except ImportError:
    HAS_MASKABLE_PPO = False
```

**位置2**：在 `ENV_TO_MODEL` 字典中添加配置（第75行之后）

```python
'TicTacToe-MaskablePPO-v0': {
    'model': MaskablePPO,
    'kwargs': {
        'policy': 'MlpPolicy',
        'learning_rate': 1e-3,
        'n_steps': 128,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'policy_kwargs': {
            'net_arch': [128, 128]
        }
    }
}
```

#### 任务2：创建测试脚本

**新文件**：`test_maskable_ppo_viper.py`（见[技术细节](#测试脚本完整代码)）

#### 任务3：运行训练和验证

```bash
# 1. 快速测试（2分钟）
python3 train/train_delta_selfplay_ppo.py \
    --total-timesteps 20000 \
    --n-env 4 \
    --output log/test_ppo.zip

# 2. 检查日志（关键指标）
# 预期：ep_rew_mean 应该在 [-1, 1] 范围（无 -10）
# 预期：非法移动 = 0

# 3. 完整训练（20分钟）
python3 train/train_delta_selfplay_ppo.py \
    --total-timesteps 200000 \
    --n-env 8 \
    --use-minmax \
    --output log/oracle_TicTacToe_ppo.zip

# 4. 评估质量
python3 evaluate_nn_quality.py --model log/oracle_TicTacToe_ppo.zip

# 5. 测试 VIPER 兼容性
python3 test_maskable_ppo_viper.py

# 6. 提取决策树
python3 main.py \
    --train-viper \
    --env-name TicTacToe-MaskablePPO-v0 \
    --oracle log/oracle_TicTacToe_ppo.zip
```

---

## 技术细节

### MaskablePPO 工作原理

```python
# 1. 环境包装
env = TicTacToeDeltaSelfPlayEnv(...)
env = ActionMasker(env, mask_fn)  # 添加 mask wrapper

def mask_fn(env):
    """返回合法动作的 mask"""
    board = env.board
    return (board == 0).astype(np.int8)  # 0=空位=合法

# 2. 训练
model = MaskablePPO('MlpPolicy', env)
model.learn(total_timesteps=200000)  # mask 自动生效

# 3. 预测（VIPER 调用方式）
action, state = model.predict(obs, deterministic=True)
# ActionMasker 自动提供 mask，policy 自动应用
```

### 为什么 PPO 不会绕过 mask？

```python
# PPO 的动作选择（简化）
logits = policy_network(obs)           # 得到每个动作的分数
masked_logits = logits.masked_fill(    # 将非法动作的分数设为 -inf
    ~action_mask, float('-inf')
)
probs = softmax(masked_logits)         # 非法动作的概率 = 0
action = sample(probs)                  # 永远不会采样到非法动作

# 对比 DQN 的 ε-greedy
if random() < epsilon:
    action = random_choice(all_actions)  # ❌ 可能选到非法动作
```

### MaskablePPO vs MaskedDQN 对比表

| 特性 | MaskablePPO | MaskedDQN (自定义) |
|------|------------|-------------------|
| 非法移动问题 | ✅ 解决 | ❌ 需重写 collect_rollouts |
| VIPER 兼容性 | ✅ 验证通过 | ✅ 理论兼容 |
| 实现复杂度 | ⭐ 低 | ⭐⭐⭐⭐ 高 |
| 依赖 | sb3-contrib | stable-baselines3 |
| 训练速度 | 中等 | 快 |
| 调试难度 | 低 | 高 |

### 测试脚本完整代码

```python
"""
test_maskable_ppo_viper.py
测试 MaskablePPO 与 VIPER 的兼容性
"""

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import gym_env

def mask_fn(env):
    """返回 action mask"""
    if hasattr(env, 'board'):
        board = env.board
    else:
        board = env.env.board
    return (board == 0).astype(np.int8)

print("=" * 70)
print("测试 MaskablePPO 与 VIPER 兼容性")
print("=" * 70)

# 1. 创建环境
print("\n1. 创建环境...")
from gym_env.tictactoe_delta_selfplay import TicTacToeDeltaSelfPlayEnv
from gym_env.policies.baseline_policies import RandomPlayerPolicy

obs_space = gym.spaces.Box(low=-1, high=1, shape=(9,), dtype=np.float32)
act_space = gym.spaces.Discrete(9)
baseline_pool = [RandomPlayerPolicy(obs_space, act_space)]

env = TicTacToeDeltaSelfPlayEnv(
    baseline_pool=baseline_pool,
    learned_pool=None
)
env = ActionMasker(env, mask_fn)
print("✓ 环境创建成功")

# 2. 创建模型
print("\n2. 创建 MaskablePPO 模型...")
model = MaskablePPO(
    policy='MlpPolicy',
    env=env,
    learning_rate=1e-3,
    n_steps=128,
    batch_size=64,
    verbose=1
)
print("✓ 模型创建成功")

# 3. 短暂训练
print("\n3. 训练 5000 步...")
model.learn(total_timesteps=5000)
print("✓ 训练完成")

# 4. 测试 predict() 接口（模拟 VIPER 调用）
print("\n4. 测试 VIPER 兼容的 predict() 接口...")
obs, _ = env.reset()

# 模拟 VIPER 调用（不提供 action_masks）
action, state = model.predict(obs, deterministic=True)
print(f"  ✓ predict(obs, deterministic=True) 成功")
print(f"    返回: action={action}, state={state}")

# 5. 检查非法移动
print("\n5. 测试 100 局，检查非法移动...")
illegal_count = 0
total_reward = 0

for _ in range(100):
    obs, _ = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        done = terminated or truncated

        if done and 'illegal_move' in info and info['illegal_move']:
            illegal_count += 1

    total_reward += episode_reward

avg_reward = total_reward / 100

print(f"\n结果:")
print(f"  非法移动次数: {illegal_count}/100")
print(f"  平均奖励: {avg_reward:.2f}")

if illegal_count == 0:
    print("\n✅ 测试通过！")
    print("  - 无非法移动")
    print("  - MaskablePPO 正常工作")
    print("  - 与 VIPER 接口兼容")
else:
    print(f"\n⚠ 警告：有 {illegal_count} 次非法移动")

env.close()
```

### 环境与 VIPER 的集成点

```python
# train/viper.py:157
action, _states = active_policy.predict(obs, deterministic=True)

# MaskablePPO 完全兼容这个接口
# action_masks 参数是可选的，不提供时：
# - 如果环境有 ActionMasker wrapper → 自动获取 mask
# - 如果没有 wrapper → 不使用 mask（但不会报错）
```

---

## 预期结果

### 训练日志（正常情况）

```
----------------------------------
| rollout/            |          |
|    ep_len_mean      | 3.2      |
|    ep_rew_mean      | -0.3     |  ⭐ 应该接近 0，不是 -6.18
| time/               |          |
|    episodes         | 600      |
----------------------------------

测试结果 (50局 vs MinMax):
  胜: 2 (4.0%)
  负: 3 (6.0%)
  平: 45 (90.0%)
  非法移动: 0  ⭐ 关键指标
```

### 评估输出（最优策略）

```
✓ Action Masking 功能正常工作
✓ 关键局面识别: 100.0%
✓ vs MinMax平局率: 98.0%
✓ 无输局

🏆 恭喜！你的神经网络已达到TicTacToe的最优策略！
```

---

## FAQ

### Q1: 为什么不继续用 DQN？
**A**: DQN 的 ε-greedy exploration 会绕过 mask。要解决需要重写 `collect_rollouts()`，工作量大且容易出错。

### Q2: MaskablePPO 训练会慢多少？
**A**: 比 DQN 慢约 20-30%，但比自己调试 MaskedDQN 快得多。TicTacToe 规模小，速度差异不明显。

### Q3: 如何确认 mask 生效了？
**A**: 看两个指标：
1. 训练日志：`ep_rew_mean` 应该在 [-1, 1]，不应该有 -10
2. 测试结果：`非法移动: 0`

### Q4: ActionMasker 会影响 VIPER 吗？
**A**: 不会。VIPER 调用 `predict(obs, deterministic=True)` 时，ActionMasker 只是在内部提供 mask，不改变外部接口。

### Q5: 为什么 sb3-contrib 要用 2.4.0？
**A**: 与你环境中的 stable-baselines3==1.5.0 版本匹配。太高版本可能不兼容。

### Q6: 如果 MaskablePPO 也失败怎么办？
**A**: 回退到方案B（自定义 MaskedDQN）。文件已创建：[gym_env/masked_dqn.py](gym_env/masked_dqn.py)（虽然被拒绝但可以恢复）。

### Q7: 环境的 -10 惩罚还保留吗？
**A**: 是的，保留在 [tictactoe_delta_selfplay.py:144](gym_env/tictactoe_delta_selfplay.py#L144)。作为安全网，如果 mask 失效会暴露问题。

---

## 关键文件清单

### 需要修改的文件
- [ ] [train/oracle.py](train/oracle.py) - 添加 MaskablePPO 配置
- [ ] 新建 `test_maskable_ppo_viper.py` - 测试脚本

### 可以直接使用的文件
- ✅ [train/train_delta_selfplay_ppo.py](train/train_delta_selfplay_ppo.py)
- ✅ [requirements.txt](requirements.txt)
- ✅ [evaluate_nn_quality.py](evaluate_nn_quality.py)

### 参考文件（已实现但有问题）
- [gym_env/masked_dqn_policy.py](gym_env/masked_dqn_policy.py) - MaskedDQNPolicy（有 exploration 问题）
- [train/train_delta_selfplay.py](train/train_delta_selfplay.py) - DQN 训练脚本（非法移动率高）

---

## 进度追踪

- [x] 分析问题根因（DQN ε-greedy 绕过 mask）
- [x] 验证 MaskablePPO 与 VIPER 兼容性
- [x] 更新依赖文件
- [ ] 修改 oracle.py
- [ ] 创建测试脚本
- [ ] 运行完整训练
- [ ] 验证无非法移动
- [ ] VIPER 决策树提取
- [ ] 文档完善

---

**当前状态**: 方案已确定，等待实施
**推荐方案**: MaskablePPO
**预计完成时间**: 1-2 小时（含训练）
**最后更新**: 2025-01-XX
