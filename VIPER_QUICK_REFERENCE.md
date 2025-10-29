# VIPER MaskablePPO 快速参考

## 🔧 Bug 修复

### 问题 1: `AttributeError: 'OrderEnforcing' object has no attribute 'board'`

**原因**: Gymnasium 环境包装器层级导致无法直接访问 `board` 属性。

**修复位置**: `train/viper_maskable_ppo.py` 第 41-75 行的 `mask_fn` 函数

**修复方法**: 使用递归解包
```python
def mask_fn(env):
    current_env = env
    while hasattr(current_env, 'env') and not hasattr(current_env, 'board'):
        current_env = current_env.env
    return (current_env.board == 0).astype(np.int8)
```

**状态**: ✅ 已修复

---

## ❓ 常见问题

### Q1: 为什么 VIPER 训练只用一个对手？

**答案**: 这是**正常的**，符合 VIPER 论文设计。

**原因**:
- VIPER 是**模仿学习**，不是从零训练
- Oracle (PPO) 已经在多对手上训练过
- VIPER 只需提取 Oracle 学到的知识
- 单一对手提供稳定的状态分布

**详细解释**: 见 [VIPER_TECHNICAL_ANALYSIS.md](VIPER_TECHNICAL_ANALYSIS.md) 第 3.2 节

### Q2: 本实现与 VIPER 原论文有什么区别？

**核心算法**: ✅ 完全一致（DAgger + Weighted Imitation Learning）

**主要区别**: 针对 TicTacToe 的必要适配

| 维度 | 原论文 | 本实现 | 原因 |
|------|--------|--------|------|
| Oracle | DQN/PPO | MaskablePPO | 支持 action masking |
| Criticality | 所有动作 | 仅合法动作 | 避免非法动作污染 |
| 树输出 | 单个动作 | 概率+Masking | 保证 100% 合法 |

**结论**: 保持算法一致性，仅做必要适配

---

## 📊 训练参数快速选择

### 标准训练（推荐）⭐

```bash
python train/viper_maskable_ppo.py \
    --oracle-path log/oracle_TicTacToe_ppo_aggressive.zip \
    --output log/viper_from_ppo.joblib \
    --total-timesteps 50000 \
    --n-iter 10 \
    --max-depth 10 \
    --max-leaves 50 \
    --opponent-type minmax \
    --test
```

**预计时间**: 10-15 分钟

### 快速测试

```bash
python train/viper_maskable_ppo.py \
    --oracle-path log/oracle_TicTacToe_ppo_aggressive.zip \
    --output log/viper_quick.joblib \
    --total-timesteps 20000 \
    --n-iter 5 \
    --max-depth 8 \
    --max-leaves 30 \
    --opponent-type minmax \
    --test
```

**预计时间**: 5 分钟

### 高质量训练

```bash
python train/viper_maskable_ppo.py \
    --oracle-path log/oracle_TicTacToe_ppo_aggressive.zip \
    --output log/viper_high_quality.joblib \
    --total-timesteps 100000 \
    --n-iter 15 \
    --max-depth 12 \
    --max-leaves 80 \
    --opponent-type minmax \
    --test
```

**预计时间**: 20-30 分钟

---

## 📈 性能期望

### 对战 MinMax（最优对手）

| 性能等级 | 平局率 | 非法移动 |
|----------|--------|----------|
| ✅ 优秀 | ≥ 80% | 0 |
| △ 良好 | 60-80% | 0 |
| ✗ 需改进 | < 60% | 0 |

**说明**: MinMax 是最优策略，平局率高说明学到了接近最优的决策。

### 对战 Random

| 性能等级 | 胜率 | 非法移动 |
|----------|------|----------|
| ✅ 优秀 | ≥ 90% | 0 |
| △ 良好 | 70-90% | 0 |
| ✗ 需改进 | < 70% | 0 |

**重要**: 非法移动数必须为 0，否则说明 masking 失败。

---

## 🔍 评估命令

### 基本评估

```bash
python evaluation/evaluate_viper_tree.py \
    --model-path log/viper_from_ppo.joblib \
    --opponent both \
    --n-episodes 100
```

### 导出规则

```bash
python evaluation/evaluate_viper_tree.py \
    --model-path log/viper_from_ppo.joblib \
    --export-rules log/tree_rules.txt
```

### 可视化决策

```bash
python evaluation/evaluate_viper_tree.py \
    --model-path log/viper_from_ppo.joblib \
    --visualize
```

---

## 🎯 参数调优指南

### 如果性能不佳

**方案 1: 增加数据量**
```bash
--total-timesteps 100000  # 从 50000 增加
```

**方案 2: 增加迭代次数**
```bash
--n-iter 15  # 从 10 增加
```

**方案 3: 增加树复杂度**
```bash
--max-depth 12  # 从 10 增加
--max-leaves 80  # 从 50 增加
```

**方案 4: 使用更强的 Oracle**
- 训练 PPO 更多步数
- 提高探索性（`--ent-coef 0.05`）

### 如果树太复杂（可解释性差）

**方案 1: 限制树大小**
```bash
--max-depth 6   # 从 10 减少
--max-leaves 20  # 从 50 减少
```

**方案 2: 减少数据量**
```bash
--total-timesteps 30000  # 避免过拟合
```

---

## 📝 核心算法流程

```
┌─────────────────────────────────────────────┐
│          VIPER 训练流程图                    │
└─────────────────────────────────────────────┘

1. 加载 MaskablePPO Oracle
   ↓
2. D ← ∅  (空数据集)
   ↓
3. FOR iter = 1 to N:
   │
   ├─→ 3.1 Beta 调度
   │       β = 1 (iter 0) or 0 (iter 1+)
   │
   ├─→ 3.2 采样轨迹
   │       if β == 1:
   │         用 Oracle 采样 (高质量初始数据)
   │       else:
   │         用 Tree 采样 (DAgger 风格)
   │
   ├─→ 3.3 计算 Criticality
   │       weight = log_prob_max - log_prob_min
   │       (仅考虑合法动作)
   │
   ├─→ 3.4 聚合数据
   │       D = D ∪ new_trajectory
   │
   ├─→ 3.5 训练决策树
   │       tree.fit(X, y, sample_weight=weight)
   │
   └─→ 3.6 评估性能
           记录 reward
   │
4. 返回最佳树
```

---

## 🔑 关键技术点

### 1. Criticality Loss（重要性权重）

**公式**:
```
Criticality(s) = max_a∈Legal log π(a|s) - min_a∈Legal log π(a|s)
```

**含义**:
- 高 → 决策很重要（选对很关键）
- 低 → 随便选都行

**作用**:
重要状态在训练时权重更高

### 2. Beta 调度（DAgger）

```python
# Iteration 0
beta = 1.0  → 100% Oracle 采样

# Iteration 1+
beta = 0.0  → 100% Tree 采样，但标签仍是 Oracle 动作
```

**目的**: 修正协变量偏移（Covariate Shift）

### 3. Probability Masking

**训练**: 直接学习 Oracle 的动作标签
**推理**:
```python
probs = tree.predict_proba(obs)
legal_actions = where(obs == 0)
masked_probs = probs.copy()
masked_probs[illegal_actions] = -inf
action = argmax(masked_probs)  # 保证合法
```

---

## 📚 相关文档

| 文档 | 内容 |
|------|------|
| [VIPER_MASKABLE_PPO_GUIDE.md](VIPER_MASKABLE_PPO_GUIDE.md) | 完整使用指南 |
| [VIPER_TECHNICAL_ANALYSIS.md](VIPER_TECHNICAL_ANALYSIS.md) | 技术分析与原论文对比 |
| 本文档 | 快速参考 |

---

## 🚦 验证清单

训练完成后，检查以下指标：

- [ ] 非法移动数 = 0
- [ ] 对战 MinMax 平局率 ≥ 60%
- [ ] 对战 Random 胜率 ≥ 70%
- [ ] 树深度合理（< 15）
- [ ] 可以导出规则（`--export-rules`）
- [ ] 性能达到 Oracle 的 70%+

---

## 💡 示例完整工作流

```bash
# 1. 检查 Oracle
ls -lh log/oracle_TicTacToe_ppo*.zip

# 2. 训练 VIPER（标准参数）
python train/viper_maskable_ppo.py \
    --oracle-path log/oracle_TicTacToe_ppo_aggressive.zip \
    --output log/viper_from_ppo.joblib \
    --total-timesteps 50000 \
    --n-iter 10 \
    --max-depth 10 \
    --max-leaves 50 \
    --opponent-type minmax \
    --test \
    --verbose 1

# 3. 详细评估
python evaluation/evaluate_viper_tree.py \
    --model-path log/viper_from_ppo.joblib \
    --opponent both \
    --n-episodes 100 \
    --export-rules log/tree_rules.txt \
    --visualize

# 4. 查看规则
cat log/tree_rules.txt
```

**完成！** 🎉
