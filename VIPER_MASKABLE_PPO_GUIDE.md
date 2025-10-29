# VIPER for MaskablePPO - 使用指南

本指南说明如何使用 MaskablePPO 训练的神经网络模型，通过 VIPER 框架训练一个可解释的决策树。

## 📋 概述

**目标**：将通过 MaskablePPO 训练的 TicTacToe 神经网络策略，提取为一个可解释的决策树。

**方法**：
1. ✅ 使用单棵分类树（最高可解释性）
2. ✅ 输出每个动作的概率分布（predict_proba）
3. ✅ 使用 action masking 避免非法移动
4. ✅ 计算 criticality loss 作为样本权重

**优势**：
- 🎯 完整的 IF-THEN 规则（可人类理解）
- 🚫 100% 避免非法移动
- 📊 更小的模型（vs 9棵回归树）
- 🔍 可提取和验证决策规则

---

## ⚠️ 重要说明

**Bug 修复**: 如果遇到 `AttributeError: 'OrderEnforcing' object has no attribute 'board'` 错误，说明 `mask_fn` 函数需要更新。已在最新代码中修复（使用递归解包环境层级）。

**对手配置**: VIPER 训练默认使用**单一对手**（MinMax 或 Random），这是符合 VIPER 论文设计的。详见 [技术分析文档](VIPER_TECHNICAL_ANALYSIS.md)。

---

## 🚀 快速开始

### 步骤 1: 训练 MaskablePPO Oracle（如果还没有）

如果你已经有训练好的模型（如 `log/oracle_TicTacToe_ppo_aggressive.zip`），可以跳过这步。

```bash
# 使用 delta self-play 训练 MaskablePPO
python train/train_delta_selfplay_ppo.py \
    --total-timesteps 200000 \
    --n-env 8 \
    --update-interval 10000 \
    --max-pool-size 20 \
    --play-as-o-prob 0.5 \
    --output log/oracle_TicTacToe_ppo_aggressive.zip \
    --ent-coef 0.05 \
    --random-weight 2.0 \
    --use-minmax
```

**参数说明**：
- `--total-timesteps`: 总训练步数
- `--n-env`: 并行环境数
- `--ent-coef`: 熵系数（控制探索性）
- `--random-weight`: Random 对手的采样权重
- `--use-minmax`: 是否加入 MinMax 对手

---

### 步骤 2: 使用 VIPER 训练决策树

使用训练好的 MaskablePPO 模型作为 Oracle，训练决策树：

```bash
python train/viper_maskable_ppo.py \
    --oracle-path log/oracle_TicTacToe_ppo_aggressive.zip \
    --output log/viper_TicTacToe_from_ppo.joblib \
    --total-timesteps 50000 \
    --n-iter 10 \
    --max-depth 10 \
    --max-leaves 50 \
    --opponent-type minmax \
    --test \
    --verbose 1
```

**参数说明**：
- `--oracle-path`: MaskablePPO 模型路径（必需）
- `--output`: 输出决策树路径
- `--total-timesteps`: VIPER 采样总步数
- `--n-iter`: VIPER 迭代次数（每次迭代训练一棵树，选最好的）
- `--max-depth`: 决策树最大深度
- `--max-leaves`: 决策树最大叶子节点数
- `--opponent-type`: 训练时的对手类型（`random` 或 `minmax`）
- `--test`: 训练完成后自动测试
- `--verbose`: 详细程度（0=安静，1=正常，2=调试）

**推荐参数组合**：

| 场景 | total_timesteps | n_iter | max_depth | max_leaves |
|------|----------------|--------|-----------|------------|
| 快速测试 | 20000 | 5 | 8 | 30 |
| 标准训练 | 50000 | 10 | 10 | 50 |
| 高质量模型 | 100000 | 15 | 12 | 80 |

---

### 步骤 3: 评估决策树

```bash
# 评估对战 Random 和 MinMax
python evaluation/evaluate_viper_tree.py \
    --model-path log/viper_TicTacToe_from_ppo.joblib \
    --opponent both \
    --n-episodes 100

# 导出决策规则到文本文件
python evaluation/evaluate_viper_tree.py \
    --model-path log/viper_TicTacToe_from_ppo.joblib \
    --opponent minmax \
    --n-episodes 100 \
    --export-rules log/tree_rules.txt

# 可视化决策过程
python evaluation/evaluate_viper_tree.py \
    --model-path log/viper_TicTacToe_from_ppo.joblib \
    --opponent minmax \
    --n-episodes 100 \
    --visualize
```

---

## 📊 评估指标

### TicTacToe 性能标准

**对战 MinMax (最优对手)**：
- ✅ **优秀**: 平局率 ≥ 80%（说明学到了接近最优策略）
- △ **良好**: 平局率 60-80%
- ✗ **需改进**: 平局率 < 60%

**对战 Random**：
- ✅ **优秀**: 胜率 ≥ 90%
- △ **良好**: 胜率 70-90%
- ✗ **需改进**: 胜率 < 70%

**非法移动**：
- ✅ **必须**: 非法移动数 = 0（如果有非法移动，说明 masking 失败）

---

## 🔧 参数调优建议

### 如果决策树性能不佳：

1. **增加采样数据量**
   ```bash
   --total-timesteps 100000  # 从 50000 增加到 100000
   ```

2. **增加迭代次数**
   ```bash
   --n-iter 15  # 从 10 增加到 15
   ```

3. **调整树的复杂度**
   ```bash
   --max-depth 12      # 增加深度
   --max-leaves 80     # 增加叶子节点
   ```

4. **使用更强的 Oracle**
   - 训练 PPO 更多步数
   - 提高 `--ent-coef`（增加探索）
   - 使用 MinMax 作为对手

5. **改变训练对手**
   ```bash
   --opponent-type random  # 如果对 minmax 训练效果不好，先用 random
   ```

### 如果树太复杂（可解释性差）：

1. **限制树的大小**
   ```bash
   --max-depth 6       # 减少深度
   --max-leaves 20     # 减少叶子节点
   ```

2. **减少采样数据**
   ```bash
   --total-timesteps 30000  # 减少数据量，避免过拟合
   ```

---

## 📁 文件说明

### 核心文件

| 文件 | 说明 |
|------|------|
| `train/viper_maskable_ppo.py` | VIPER 训练脚本（支持 MaskablePPO） |
| `evaluation/evaluate_viper_tree.py` | 决策树评估脚本 |
| `train/train_delta_selfplay_ppo.py` | MaskablePPO 训练脚本 |

### 输出文件

| 文件 | 说明 |
|------|------|
| `log/oracle_TicTacToe_ppo_*.zip` | 训练好的 MaskablePPO 模型 |
| `log/viper_TicTacToe_from_ppo.joblib` | 提取的决策树模型 |
| `log/tree_rules.txt` | 导出的决策规则（文本格式） |

---

## 🔍 核心技术细节

### 1. Criticality Loss（关键性损失）

用于衡量状态的"重要性"，即在该状态下选择最佳动作 vs 最差动作的差异。

```python
Q(s, a) ≈ log π(a|s)  (max entropy formulation)
Criticality(s) = max_a Q(s,a) - min_a Q(s,a) (仅考虑合法动作)
```

权重越高的样本，在训练决策树时越重要。

### 2. Action Masking

```python
legal_actions = where(board == 0)  # 空位置
masked_probs = probs.copy()
masked_probs[illegal_actions] = -inf
action = argmax(masked_probs)
```

### 3. Beta 采样策略

- **Iteration 0**: `beta = 1.0` → 100% 使用 Oracle（收集高质量数据）
- **Iteration 1+**: `beta = 0.0` → 100% 使用 Tree（DAgger 风格）

### 4. 单棵分类树 vs 多棵回归树

| 方案 | 优点 | 缺点 |
|------|------|------|
| **单棵分类树** | ✅ 完整可解释性<br>✅ 模型小<br>✅ 易提取规则 | △ 可能精度略低 |
| **多棵回归树** | ✅ 精度可能更高 | ✗ 可解释性差<br>✗ 模型大<br>✗ 难提取规则 |

本实现选择**单棵分类树 + 概率掩码**，优先保证可解释性。

---

## 🐛 常见问题

### Q1: 训练时出现非法移动怎么办？

**A**: 这不应该发生，因为：
1. Oracle (MaskablePPO) 使用 ActionMasker
2. Tree 的 `ProbabilityMaskedTreeWrapper` 内置 masking

如果出现，检查：
- `mask_fn` 是否正确实现
- 环境是否正确返回 board 状态

### Q2: 决策树性能远低于神经网络怎么办？

**A**: 这是正常的，因为：
1. 决策树的表达能力比神经网络弱
2. 追求可解释性需要牺牲一定性能

改进方法：
- 增加 `--total-timesteps` 和 `--n-iter`
- 增加 `--max-depth` 和 `--max-leaves`
- 使用更强的 Oracle（训练更久的 PPO）

### Q3: 如何提取和理解决策规则？

**A**: 使用评估脚本导出规则：

```bash
python evaluation/evaluate_viper_tree.py \
    --model-path log/viper_TicTacToe_from_ppo.joblib \
    --export-rules log/tree_rules.txt
```

规则格式：
```
|--- pos_0 <= 0.50
|   |--- pos_4 <= 0.50
|   |   |--- class: 4  (center)
|   |--- pos_4 >  0.50
|   |   |--- class: 0  (corner)
...
```

解读：
- `pos_0` 到 `pos_8` 对应棋盘位置（0-8）
- 值：`-1` = 对手棋子(O), `0` = 空, `1` = 我方棋子(X)
- `class` = 选择的动作（0-8）

### Q4: 可以用于其他环境吗？

**A**: 需要修改：
1. `ProbabilityMaskedTreeWrapper` 的 `n_actions`
2. `mask_fn` 的实现（根据环境返回合法动作）
3. `get_criticality_loss_maskable_ppo` 的 action space 检查

---

## 📚 参考资料

- **VIPER 论文**: [Verifiable Reinforcement Learning via Policy Extraction](https://arxiv.org/abs/1805.08328)
- **MaskablePPO**: [sb3-contrib documentation](https://sb3-contrib.readthedocs.io/)
- **TicTacToe 环境**: `gym_env/tictactoe.py`

---

## 💡 下一步

1. **可视化决策树**：使用 `sklearn.tree.plot_tree` 或 `dtreeviz`
2. **形式化验证**：使用提取的规则进行形式化验证
3. **规则简化**：使用 `model/rule_extractor.py` 简化规则
4. **部署**：决策树可以直接转换为代码，无需依赖深度学习框架

---

## 🎯 示例工作流

```bash
# 1. 训练 MaskablePPO Oracle (如果没有)
python train/train_delta_selfplay_ppo.py \
    --total-timesteps 200000 \
    --output log/oracle_TicTacToe_ppo_aggressive.zip

# 2. 使用 VIPER 提取决策树
python train/viper_maskable_ppo.py \
    --oracle-path log/oracle_TicTacToe_ppo_aggressive.zip \
    --output log/viper_tree.joblib \
    --total-timesteps 50000 \
    --n-iter 10 \
    --test

# 3. 评估决策树
python evaluation/evaluate_viper_tree.py \
    --model-path log/viper_tree.joblib \
    --opponent both \
    --export-rules log/tree_rules.txt \
    --visualize

# 4. 查看规则
cat log/tree_rules.txt
```

完成！你现在有了一个可解释的决策树模型。🎉
