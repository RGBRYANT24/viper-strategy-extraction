# 对称性改进指南

## 对称性是否重要？

### TL;DR
**不是必需的，但有益。** 你的模型战术知识 78%、实战表现优秀，即使对称性为 0 也能打好。

### 对称性的意义

#### 优点
1. **泛化能力**: 理解棋局结构本质，而非死记硬背
2. **数据效率**: 相同数据下学到更好的策略
3. **理论最优**: TicTacToe 最优策略应该是旋转/镜像对称的

#### 缺点
1. **不影响实战**: 战术知识 78% 已经很好，对称性低不影响胜率
2. **训练代价**: 需要数据增强或特殊架构，增加训练复杂度
3. **PPO 天然不对称**: 从随机初始化开始，没有内置对称性约束

### 为什么你的模型对称性差？

查看你的动作价值分析：
```
空棋盘: 4(0.27), 0(0.26), 6(0.12)  # 差距很小，说明网络记住了绝对位置
X在角落: 2(0.34), 6(0.33), 3(0.22)  # 应该对称但不对称
```

**原因:**
1. 输入编码是绝对位置 (0-8)，不是相对的
2. 没有数据增强 (旋转/镜像)
3. 网络架构是普通 MLP，不是旋转不变的

## 如何提升对称性？

### 方法 1: 数据增强 (最简单，推荐)

训练时对每个样本应用随机旋转/镜像。

**修改环境包装器:**

```python
import numpy as np
import gymnasium as gym

class SymmetricAugmentationWrapper(gym.Wrapper):
    """
    对称数据增强包装器
    随机应用旋转/镜像变换
    """
    def __init__(self, env, augment_prob=0.5):
        super().__init__(env)
        self.augment_prob = augment_prob
        self.current_transform = None
        self.current_inv_transform = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # 随机选择变换
        if np.random.random() < self.augment_prob:
            self.current_transform = np.random.choice([
                'identity', 'rot90', 'rot180', 'rot270',
                'flip_h', 'flip_v', 'flip_d1', 'flip_d2'
            ])
        else:
            self.current_transform = 'identity'

        # 应用变换
        obs = self._apply_transform(obs, self.current_transform)
        return obs, info

    def step(self, action):
        # 反变换动作（从增强空间到原始空间）
        original_action = self._inverse_transform_action(action, self.current_transform)

        # 在原始空间执行
        obs, reward, terminated, truncated, info = self.env.step(original_action)

        # 变换观察
        obs = self._apply_transform(obs, self.current_transform)

        return obs, reward, terminated, truncated, info

    def _apply_transform(self, obs, transform):
        """应用变换到观察"""
        board = obs.reshape(3, 3)

        if transform == 'identity':
            pass
        elif transform == 'rot90':
            board = np.rot90(board, k=1)
        elif transform == 'rot180':
            board = np.rot90(board, k=2)
        elif transform == 'rot270':
            board = np.rot90(board, k=3)
        elif transform == 'flip_h':
            board = np.fliplr(board)
        elif transform == 'flip_v':
            board = np.flipud(board)
        elif transform == 'flip_d1':
            board = board.T  # 主对角线翻转
        elif transform == 'flip_d2':
            board = np.fliplr(board.T)  # 副对角线翻转

        return board.flatten()

    def _inverse_transform_action(self, action, transform):
        """反变换动作"""
        row, col = action // 3, action % 3

        if transform == 'identity':
            pass
        elif transform == 'rot90':
            row, col = col, 2 - row
        elif transform == 'rot180':
            row, col = 2 - row, 2 - col
        elif transform == 'rot270':
            row, col = 2 - col, row
        elif transform == 'flip_h':
            col = 2 - col
        elif transform == 'flip_v':
            row = 2 - row
        elif transform == 'flip_d1':
            row, col = col, row
        elif transform == 'flip_d2':
            row, col = 2 - col, 2 - row

        return row * 3 + col
```

**使用方法:**

修改 `train_delta_selfplay_ppo.py` 中的 `make_masked_env`:

```python
def make_masked_env():
    env = WeightedSelfPlayEnv(
        baseline_pool=baseline_policies,
        learned_pool=learned_policy_pool,
        play_as_o_prob=args.play_as_o_prob,
        sampling_strategy='uniform',
        random_weight=args.random_weight
    )
    env = Monitor(env)

    # ⭐ 添加对称增强
    env = SymmetricAugmentationWrapper(env, augment_prob=0.5)

    env = ActionMasker(env, mask_fn)
    return env
```

**优点:**
- ✅ 实现简单，不改变网络结构
- ✅ 增加数据多样性
- ✅ 强制网络学习对称策略

**缺点:**
- ⚠️ 训练速度略慢（需要应用变换）
- ⚠️ 可能需要更多训练步数收敛

### 方法 2: 对称集成 (Ensemble)

训练时不改变，但预测时对所有对称变换求平均。

```python
def predict_with_symmetry(model, obs, action_masks=None):
    """
    对称集成预测
    对所有 8 种对称变换求平均
    """
    transforms = [
        'identity', 'rot90', 'rot180', 'rot270',
        'flip_h', 'flip_v', 'flip_d1', 'flip_d2'
    ]

    action_probs_sum = np.zeros(9)

    for transform in transforms:
        # 变换观察
        transformed_obs = apply_transform(obs, transform)

        # 预测
        if action_masks is not None:
            transformed_mask = apply_transform(action_masks.reshape(3, 3), transform).flatten()
            action, _ = model.predict(transformed_obs, action_masks=transformed_mask)
        else:
            action, _ = model.predict(transformed_obs)

        # 获取动作概率分布
        probs = get_action_probs(model, transformed_obs, transformed_mask)

        # 反变换概率
        inv_probs = inverse_transform_probs(probs, transform)

        action_probs_sum += inv_probs

    # 平均并选择最优动作
    action_probs_avg = action_probs_sum / len(transforms)
    action = np.argmax(action_probs_avg)

    return action
```

**优点:**
- ✅ 不需要重新训练
- ✅ 立即提升对称性

**缺点:**
- ⚠️ 预测速度慢 8 倍
- ⚠️ 不适合实时对战

### 方法 3: 对称网络架构

使用 Group Equivariant CNNs (e.g., E(2)-CNN)。

**不推荐** 因为：
- 对 TicTacToe 这样小的问题过于复杂
- 需要大幅修改代码
- PPO + sb3_contrib 不直接支持

## 推荐方案

### 如果你在意对称性：

**阶段 1: 数据增强训练**
```bash
# 添加对称增强包装器后训练
python train/train_delta_selfplay_ppo.py \
    --total-timesteps 300000 \
    --use-minmax \
    --ent-coef 0.05 \
    --random-weight 2.0 \
    --output log/oracle_TicTacToe_ppo_symmetric.zip
```

**阶段 2: 评估对称性**
```bash
python train/evaluate_ppo_strategy.py \
    --model log/oracle_TicTacToe_ppo_symmetric.zip
```

期望：对称性准确率从 0% → 80%+

### 如果你不在意对称性：

**继续当前训练，不做修改。**

理由：
- 战术知识 78% 已经足够好
- 实战胜率优秀
- 对称性低不影响实际性能
- 添加对称性会增加训练时间和复杂度

## 先手 vs 后手测试

使用新创建的测试脚本：

```bash
python train/test_first_second_player.py \
    --model log/oracle_TicTacToe_ppo_balanced.zip \
    --num-games 100
```

这会测试：
1. ✅ 先手(X) vs Random
2. ✅ 先手(X) vs MinMax
3. ✅ 后手(O) vs Random
4. ✅ 后手(O) vs MinMax
5. ✅ 自己 vs 自己 (X vs O)

**关键检查点:**
- 先手 vs Random: 期望 >95% 胜率
- 后手 vs Random: 期望 >80% 胜率
- 先手 vs MinMax: 期望 >80% 平局率
- 后手 vs MinMax: 期望 >60% 平局率
- 自己 vs 自己: 期望 >70% 平局率 或 先后手胜率差 <10%

**如果先后手差距大 (>20%):**

可能原因：
- `--play-as-o-prob` 不是 0.5，导致训练偏向某一方
- 后手视角翻转有 bug

解决方案：
```bash
# 确保训练时先后手均衡
python train/train_delta_selfplay_ppo.py \
    --play-as-o-prob 0.5 \
    ...
```

## 总结

### 关于对称性

| 方法 | 难度 | 效果 | 推荐度 |
|------|------|------|--------|
| 数据增强 | 中 | 高 | ⭐⭐⭐⭐ |
| 对称集成 | 低 | 中 | ⭐⭐⭐ |
| 特殊架构 | 高 | 高 | ⭐ |
| 不做处理 | - | - | ⭐⭐⭐⭐⭐ (当前已足够好) |

### 关于先手/后手

1. **一定要测试**: 使用 `test_first_second_player.py`
2. **期望结果**: 先后手能力应该相近（胜率差 <20%）
3. **如果不均衡**: 检查 `--play-as-o-prob` 是否为 0.5

### 我的建议

基于你的测试结果（战术知识 78%，实战表现优秀）：

1. ✅ **先运行先手/后手测试**，确认模型均衡性
2. ⚠️ **对称性可以忽略**，不是性能瓶颈
3. ✅ 如果想要完美，可以尝试数据增强，但期望收益有限

你当前的重点应该是：
- 提升战术知识准确率（78% → 90%+）
- 确保先后手均衡
- 而不是追求对称性

对称性是"完美主义"，不是"实用主义"。
