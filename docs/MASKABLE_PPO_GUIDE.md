# MaskablePPO 使用指南

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install sb3-contrib
```

### 2. 训练模型

```bash
# 基础训练（推荐配置）
python train/train_delta_selfplay_ppo.py \
    --total-timesteps 250000 \
    --max-pool-size 5 \
    --use-minmax \
    --output log/oracle_TicTacToe_ppo_masked.zip

# 快速测试（5万步）
python train/train_delta_selfplay_ppo.py \
    --total-timesteps 50000 \
    --max-pool-size 3 \
    --use-minmax \
    --output log/oracle_TicTacToe_ppo_test.zip

# 完整训练（40万步，大网络）
python train/train_delta_selfplay_ppo.py \
    --total-timesteps 400000 \
    --max-pool-size 5 \
    --use-minmax \
    --net-arch "256,256" \
    --output log/oracle_TicTacToe_ppo_big.zip
```

### 3. 评估模型

```bash
# 注意：evaluate_nn_quality.py 需要稍微修改才能支持 PPO
# 目前它是为 DQN 设计的

# 快速测试
python -c "
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import gymnasium as gym
import gym_env

def mask_fn(env):
    return (env.board == 0).astype(int)

model = MaskablePPO.load('log/oracle_TicTacToe_ppo_masked.zip')
env = gym.make('TicTacToe-v0', opponent_type='minmax')
env = ActionMasker(env, mask_fn)

wins, draws, losses = 0, 0, 0
for _ in range(50):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if done:
            if reward > 0: wins += 1
            elif reward < 0: losses += 1
            else: draws += 1

print(f'胜:{wins} 平:{draws} 负:{losses}')
"
```

---

## 🔍 查看源码（理解内部机制）

### 查看 Masking 的实现位置

```bash
# 1. ActionMasker 包装器（如何添加 mask 到 info）
python -c "from sb3_contrib.common.wrappers import ActionMasker; import inspect; print(inspect.getsource(ActionMasker))"

# 2. MaskableActorCriticPolicy.forward（推理时如何 mask）
python -c "from sb3_contrib.ppo_mask.policies import MaskableActorCriticPolicy; import inspect; print(inspect.getsource(MaskableActorCriticPolicy.forward))"

# 3. MaskableActorCriticPolicy.evaluate_actions（训练时如何 mask）
python -c "from sb3_contrib.ppo_mask.policies import MaskableActorCriticPolicy; import inspect; print(inspect.getsource(MaskableActorCriticPolicy.evaluate_actions))"

# 4. MaskablePPO.train（完整训练循环）
python -c "from sb3_contrib import MaskablePPO; import inspect; print(inspect.getsource(MaskablePPO.train))"

# 5. MaskablePPO.predict（推理）
python -c "from sb3_contrib import MaskablePPO; import inspect; print(inspect.getsource(MaskablePPO.predict))"
```

### 查看源码文件位置

```bash
# 找到 sb3-contrib 安装位置
python -c "from sb3_contrib import MaskablePPO; import inspect; print(inspect.getfile(MaskablePPO))"

# 查看完整目录结构
python -c "from sb3_contrib import MaskablePPO; import os; print(os.path.dirname(inspect.getfile(MaskablePPO)))" | xargs ls -la
```

---

## 📊 核心机制解释

### ActionMasker: 环境包装器

```python
# 你的代码中
def mask_fn(env):
    """返回合法动作的 mask"""
    board = env.board
    return (board == 0).astype(np.int8)  # 1=合法, 0=非法

env = ActionMasker(env, mask_fn)

# 内部机制（自动执行）
# 每次 reset() 和 step() 后：
info['action_masks'] = mask_fn(env)
# 例如: [1, 1, 1, 1, 0, 1, 1, 1, 1]
#              位置4已占用 ↑
```

### Predict: 推理时的 Masking

```python
# 你调用
action, _ = model.predict(obs, deterministic=True)

# 内部机制（自动执行）
# 1. 获取 logits (所有动作的未归一化分数)
logits = policy_network(obs)  # [2.1, 1.5, 0.3, 1.8, 0.9, ...]

# 2. 应用 mask (⭐ 关键步骤)
# 从环境 info 获取 action_masks
masks = env.get_action_masks()  # [1, 1, 1, 1, 0, 1, 1, 1, 1]

# 非法动作的 logit 设为 -inf
masked_logits = torch.where(
    masks.bool(),        # mask == 1 (合法)
    logits,              # 保留原值
    torch.tensor(-1e8)   # mask == 0 (非法) → -inf
)
# 结果: [2.1, 1.5, 0.3, 1.8, -1e8, ...]
#                            ↑ 位置4非法

# 3. 选择动作
if deterministic:
    action = torch.argmax(masked_logits)  # 永远不会选到位置4
else:
    probs = softmax(masked_logits)  # 位置4的概率 = 0
    action = sample(probs)
```

### Train: 训练时的 Masking

```python
# 训练循环（自动执行）
for epoch in range(n_epochs):
    for batch in rollout_buffer:

        # 1. 重新评估动作（带 mask）
        values, log_probs, entropy = policy.evaluate_actions(
            obs=batch.obs,
            actions=batch.actions,
            action_masks=batch.action_masks  # ⭐ 使用保存的 mask
        )

        # evaluate_actions 内部（⭐ 关键）:
        logits = policy_network(obs)
        # 应用 mask
        masked_logits = torch.where(
            action_masks.bool(),
            logits,
            torch.tensor(-1e8)
        )
        # 计算 log_prob (只对合法动作)
        distribution = Categorical(logits=masked_logits)
        log_probs = distribution.log_prob(actions)

        # 2. 计算损失
        # ⭐ 关键：log_probs 已经考虑了 mask
        # 非法动作的 log_prob = -inf (概率=0)
        ratio = exp(log_probs - old_log_probs)
        policy_loss = -mean(min(ratio * advantage, ...))

        # 3. 反向传播
        loss.backward()
        optimizer.step()
```

---

## 🆚 与 DQN 的对比

| 步骤 | DQN (无 mask) | MaskablePPO (有 mask) |
|------|---------------|----------------------|
| **环境 reset** | obs, info = env.reset() | obs, info = env.reset()<br>**+ info['action_masks']** |
| **推理** | q = q_net(obs)<br>action = argmax(q)<br>❌ 可能选非法动作 | logits = policy_net(obs)<br>**logits[mask==0] = -inf**<br>action = argmax(logits)<br>✅ 100%合法 |
| **经验保存** | buffer.add(obs, a, r, obs') | buffer.add(obs, a, r, obs', **masks**) |
| **训练目标** | Q_target = r + γ max Q(obs', a')<br>❌ max 包括非法动作 | V_target = r + γ V(obs')<br>✅ V 只评估合法动作 |
| **损失计算** | loss = (Q - Q_target)²<br>❌ 可能用非法动作的Q | loss = PPO_clip(log_prob, ...)<br>**✅ log_prob 只来自合法动作** |
| **非法动作惩罚** | reward = -10<br>❌ 污染Q值 | 不需要惩罚<br>✅ 永远不会选非法动作 |

---

## 🎯 关键优势

### ✅ 训练时 Masking

```python
# 1. Policy Gradient 只来自合法动作
# 非法动作的概率 = 0
# 梯度不会来自非法动作

# 2. Value Function 正确评估
# V(s) 反映"只选合法动作"的期望回报
# 不会被非法动作污染

# 3. 熵计算正确
# H = -Σ p(a) log p(a)  只对合法动作求和
# 鼓励在合法动作间探索
```

### ✅ 推理时 Masking

```python
# 1. 100% 保证合法
# 非法动作 logit = -inf
# softmax 后概率 = 0
# 永远不会被选中

# 2. 无需额外检查
# 不需要 if-else 判断
# 代码简洁
```

---

## 📝 常用命令总结

```bash
# ==================== 训练 ====================
# 推荐配置
python train/train_delta_selfplay_ppo.py \
    --total-timesteps 250000 \
    --max-pool-size 5 \
    --use-minmax \
    --output log/oracle_TicTacToe_ppo.zip

# ==================== 查看源码 ====================
# ActionMasker（环境包装）
python -c "from sb3_contrib.common.wrappers import ActionMasker; import inspect; print(inspect.getsource(ActionMasker))" | less

# 推理时 masking
python -c "from sb3_contrib.ppo_mask.policies import MaskableActorCriticPolicy; import inspect; print(inspect.getsource(MaskableActorCriticPolicy.forward))" | less

# 训练时 masking
python -c "from sb3_contrib.ppo_mask.policies import MaskableActorCriticPolicy; import inspect; print(inspect.getsource(MaskableActorCriticPolicy.evaluate_actions))" | less

# ==================== 快速测试 ====================
# 测试 vs MinMax
python -c "
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import gymnasium as gym
import gym_env

def mask_fn(env):
    return (env.board == 0).astype(int)

model = MaskablePPO.load('log/oracle_TicTacToe_ppo.zip')
env = gym.make('TicTacToe-v0', opponent_type='minmax')
env = ActionMasker(env, mask_fn)

wins, draws, losses = 0, 0, 0
for _ in range(50):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if done:
            if reward > 0: wins += 1
            elif reward < 0: losses += 1
            else: draws += 1

print(f'胜:{wins} 平:{draws} 负:{losses}')
"
```

---

## 💡 故障排查

### 问题1: 导入错误

```bash
# 错误: ModuleNotFoundError: No module named 'sb3_contrib'
# 解决:
pip install sb3-contrib
```

### 问题2: ActionMasker 找不到

```bash
# 错误: cannot import name 'ActionMasker'
# 检查版本:
pip show sb3-contrib
# 需要 sb3-contrib >= 1.6.0

# 升级:
pip install --upgrade sb3-contrib
```

### 问题3: action_masks 未传递

```bash
# 错误: 训练时非法动作仍然出现
# 检查:
# 1. 环境是否用 ActionMasker 包装？
env = ActionMasker(env, mask_fn)  # 必须

# 2. mask_fn 是否正确返回？
def mask_fn(env):
    return (env.board == 0).astype(np.int8)  # 必须是 int8 或 bool
```

---

## 🔗 参考资源

- **sb3-contrib 文档**: https://sb3-contrib.readthedocs.io/
- **MaskablePPO 示例**: https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html
- **源码**: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib

---

## 总结

MaskablePPO 通过以下机制实现完整的 action masking：

1. **ActionMasker** 包装器：自动添加 `action_masks` 到 `info`
2. **Policy.forward()**: 推理时 `logits[mask==0] = -inf`
3. **Policy.evaluate_actions()**: 训练时 `logits[mask==0] = -inf`
4. **PPO 损失**: 只来自合法动作的 `log_prob`

无需手动修改，开箱即用！🎉
