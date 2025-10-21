# 回归树（Q值树）使用指南

## 📋 目录
1. [核心概念](#核心概念)
2. [为什么使用回归树](#为什么使用回归树)
3. [快速开始](#快速开始)
4. [详细对比](#详细对比)
5. [实验结果](#实验结果)
6. [可解释性](#可解释性)

---

## 核心概念

### 分类树 vs 回归树

| 特性 | 分类树（当前） | 回归树（新方案） |
|------|------------|--------------|
| **输出类型** | 单个类别（动作0-8） | 9个连续值（Q值） |
| **训练目标** | 动作标签 | 每个动作的价值估计 |
| **动作选择** | 直接输出 | argmax(Q值 × 合法掩码) |
| **非法动作** | ❌ 可能输出非法动作 | ✅ 自然过滤非法动作 |
| **可解释性** | IF-THEN规则 → 动作 | IF-THEN规则 → Q值 |

### 工作原理

```python
# 分类树
board = [0, 0, 0, 0, 1, 0, 0, 0, 0]  # 中心有X
action = tree.predict(board)         # 直接输出: 3
# 问题：如果位置3已被占用？→ 非法动作！

# 回归树
board = [0, 0, 0, X, 1, 0, 0, 0, 0]  # 中心有X，位置3已占用
q_values = tree.predict(board)       # 输出: [0.2, 0.1, 0.3, 0.5, 0.4, 0.1, 0.0, 0.0, 0.0]
legal_actions = [0, 1, 2, 4, 5, 6, 7, 8]  # 位置3非法
action = argmax(q_values[legal_actions])   # 选择位置4 (Q=0.4，合法动作中最高)
# ✅ 100%避免非法动作！
```

---

## 为什么使用回归树

### ❌ 分类树的问题

1. **静态输出空间**
   - 输出固定在9个类别（0-8）
   - 不知道哪些动作在当前状态下合法
   - 即使训练得再好，未见过的局面仍可能输出非法动作

2. **无法表达次优选择**
   - 只输出"最优"动作
   - 当最优动作非法时，无法知道次优是什么

### ✅ 回归树的优势

1. **动态动作过滤**
   ```python
   # 推理时可以动态应用约束
   q_values = tree.predict(state)
   legal_q = q_values[legal_actions]  # 只看合法动作
   action = legal_actions[argmax(legal_q)]  # 100%合法
   ```

2. **保留完整偏好信息**
   ```python
   q_values = [0.1, 0.05, 0.15, 0.4, 0.2, 0.05, 0.03, 0.01, 0.01]
   # 偏好顺序: 3 > 4 > 2 > 0 > ...
   # 如果3非法 → 自动选择4（次优）
   ```

3. **更好的可解释性**
   ```
   决策解释：
   - 决策树预测的Q值:
     位置0: 0.1, 位置1: 0.05, ..., 位置3: 0.4, 位置4: 0.2
   - 位置3 Q值最高(0.4)，但已被占用
   - 在合法动作中，位置4 Q值最高(0.2)
   - 因此选择位置4
   ```

---

## 快速开始

### 前置条件检查

```bash
# 检查是否已有Oracle模型
ls log/oracle_TicTacToe-v0.zip

# 如果文件不存在，需要先训练Oracle
```

### 步骤1: 训练Oracle（如果还没有）

```bash
# 训练神经网络Oracle（首次运行需要）
python main.py train-oracle \
  --env-name TicTacToe-v0 \
  --total-timesteps 50000 \
  --verbose 1

# 预计时间：5-10分钟
# 输出文件：log/oracle_TicTacToe-v0.zip
```

### 步骤2: 训练回归树

```bash
# 使用回归树训练VIPER
python main.py train-viper-regression \
  --env-name TicTacToe-v0 \
  --n-iter 20 \
  --max-depth 15 \
  --total-timesteps 10000 \
  --verbose 1

# 参数说明：
# --n-iter 20          VIPER迭代次数（建议10-50）
# --max-depth 15       决策树最大深度（建议10-20）
# --total-timesteps    每次迭代采样的步数
# --verbose 1          显示进度信息

# 预计时间：3-5分钟（取决于n-iter）
# 输出文件：log/viper_TicTacToe-v0_all-leaves_15_regression.joblib
```

**输出示例**:
```
Training VIPER with Regression Trees on TicTacToe-v0
Using Q-value regression approach

Iteration 1/20: Reward = 0.6500 +/- 0.4775
Iteration 2/20: Reward = 0.7200 +/- 0.4495
...
Iteration 20/20: Reward = 0.8500 +/- 0.3571

Best policy: Iteration 18
Mean reward: 0.8500

=== Regression Tree Model Info ===
Model type: MultiOutputRegressor with 9 trees
Number of leaves: 42
Tree depth: 15
========================================

Regression tree saved to: log/viper_TicTacToe-v0_all-leaves_15_regression.joblib
```

### 步骤3: 测试回归树

```bash
# 快速测试：验证是否避免非法动作
python battle_regression_tree.py \
  --mode test \
  --regression-path log/viper_TicTacToe-v0_all-leaves_15_regression.joblib \
  --n-games 50

# 参数说明：
# --mode test          快速测试模式
# --regression-path    回归树模型文件路径
# --n-games 50         测试局数

# 预计时间：10-30秒
```

**期望输出**:
```
快速测试：回归树是否能避免非法动作

对战结果: 回归树 vs MinMax
======================================================================
总局数: 50

回归树 (X) 获胜: 10 局 (20.0%)
MinMax (O) 获胜: 25 局 (50.0%)
平局: 15 局 (30.0%)

非法移动总数: 0
  - 回归树 非法移动: 0
  - MinMax 非法移动: 0
======================================================================

✓ 成功：回归树在所有测试中都避免了非法动作！
```

### 步骤4: 对比分类树和回归树（可选）

如果你之前训练过分类树，可以进行对比：

```bash
# 首先检查是否有分类树模型
ls log/viper_TicTacToe-v0*.joblib

# 如果没有分类树，先训练一个（用于对比）
python main.py train-viper \
  --env-name TicTacToe-v0 \
  --n-iter 20 \
  --max-depth 10 \
  --total-timesteps 10000 \
  --verbose 1

# 预计时间：2-4分钟
# 输出文件：log/viper_TicTacToe-v0_all-leaves_10.joblib
```

```bash
# 对比测试
python battle_regression_tree.py \
  --mode compare \
  --classification-path log/viper_TicTacToe-v0_all-leaves_10.joblib \
  --regression-path log/viper_TicTacToe-v0_all-leaves_15_regression.joblib \
  --n-games 50

# 预计时间：30秒-1分钟
```

### 步骤5: 详细分析（单局游戏）

查看回归树如何通过Q值做决策：

```bash
# 单局详细测试，查看Q值预测过程
python battle_regression_tree.py \
  --mode single \
  --regression-path log/viper_TicTacToe-v0_all-leaves_15_regression.joblib

# 这会显示：
# - 每一步的棋盘状态
# - 9个位置的Q值预测
# - 哪些动作合法/非法
# - 最终选择的动作及原因

# 预计时间：5-10秒（单局）
```

**输出示例**:
```
============================================================
回归树决策解释 #1
============================================================
玩家ID: 1 (X)
原始棋盘: [0. 0. 0. 0. 0. 0. 0. 0. 0.]

所有位置的Q值预测:
  位置:      0      1      2      3      4      5      6      7      8
  Q值:   0.150  0.120  0.180  0.200  0.450  0.190  0.140  0.110  0.160
  状态:   合法   合法   合法   合法   合法   合法   合法   合法   合法

合法动作: [0 1 2 3 4 5 6 7 8]
合法动作的Q值:
  动作 0: Q = 0.150
  动作 1: Q = 0.120
  动作 2: Q = 0.180
  动作 3: Q = 0.200
  动作 4: Q = 0.450 ← 最终选择
  动作 5: Q = 0.190
  动作 6: Q = 0.140
  动作 7: Q = 0.110
  动作 8: Q = 0.160

最终选择: 动作 4 (Q = 0.450)
============================================================
```

---

## 详细对比

### 实验设置

```bash
# 设置1: 对抗MinMax（完美玩家）
回归树 vs MinMax: 100局
分类树 vs MinMax: 100局

# 设置2: 相互对战
回归树 vs 分类树: 100局

# 设置3: 边缘案例测试
特殊棋盘局面: 50种未见过的局面
```

### 预期结果

| 指标 | 分类树 | 回归树 | 改进 |
|------|--------|--------|------|
| **非法动作率** | 5-15% | 0% | ✅ 100% |
| **对MinMax胜率** | 10-20% | 15-25% | ✅ 提升 |
| **平局率** | 30-40% | 35-45% | ✅ 提升 |
| **可解释性** | 高 | 高+ | ✅ 更丰富 |

### 为什么回归树更强

1. **从不输在非法动作上**
   - 分类树：因非法动作直接判负
   - 回归树：100%避免非法动作

2. **更好的探索-利用平衡**
   - 分类树：只知道一个"最优"动作
   - 回归树：知道所有动作的价值排序

3. **对未见过局面的泛化**
   - 分类树：可能输出非法动作
   - 回归树：总能找到合法的次优动作

---

## 可解释性

### 规则提取

回归树仍然可以提取IF-THEN规则，只是输出变成Q值：

```python
# 使用规则提取器
from model.rule_extractor import DecisionTreeRuleExtractor

# 对于MultiOutputRegressor，提取第一个树的规则
base_tree = regression_model.estimators_[4]  # 位置4的Q值树

extractor = DecisionTreeRuleExtractor(
    tree_model=base_tree,
    X_train=X_train,
    y_train=Q_train[:, 4],  # 位置4的Q值
    feature_names=[f"pos_{i}" for i in range(9)]
)

rules = extractor.extract_rules(verbose=True)
```

**规则示例**:
```
规则1: IF pos_4 <= 0.5 AND pos_0 <= 0.5 AND pos_8 <= 0.5
       THEN Q(action_4) = 0.45

解释: 当中心为空(pos_4≤0.5)，左上角为空(pos_0≤0.5)，右下角为空(pos_8≤0.5)时，
     选择中心位置的价值为0.45（较高）
```

### Q值可视化

```python
def visualize_q_values(regression_tree, board_state):
    """可视化某个棋盘状态的Q值"""
    q_values = regression_tree.predict_q_values(board_state)

    # 3x3显示
    q_matrix = q_values.reshape(3, 3)
    legal_mask = (board_state == 0).reshape(3, 3)

    print("Q值热力图:")
    for i in range(3):
        for j in range(3):
            if legal_mask[i, j]:
                print(f"{q_matrix[i,j]:6.3f}", end=" ")
            else:
                print("  XXX ", end=" ")
        print()
```

---

## 常见问题

### Q1: 回归树会比分类树慢吗？

**答**：训练会稍慢（需要预测9个值），但推理速度相同。

- 训练时间：回归树约为分类树的1.5-2倍
- 推理时间：几乎相同（都是树遍历）

### Q2: 回归树的模型大小？

**答**：更大，因为是MultiOutputRegressor（9棵树）。

- 分类树：1棵树，约50-100个叶节点
- 回归树：9棵树，每棵约30-60个叶节点
- 总大小：约3-5倍

### Q3: 可以只用单棵树输出9个值吗？

**答**：可以，但效果可能不如MultiOutputRegressor。

```python
# 单树方案（需要sklearn支持多输出）
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=20)
tree.fit(X, Q)  # Q shape: (n_samples, 9)
```

### Q4: 回归树的可解释性会降低吗？

**答**：不会，反而增强了。

- 分类树：只解释"为什么选A"
- 回归树：解释"为什么A比B好，B比C好"

---

## 高级用法

### 自定义Q值提取

```python
def extract_custom_q_values(oracle, observations):
    """
    自定义Q值提取逻辑

    例如：使用UCB（Upper Confidence Bound）
    """
    raw_q = extract_q_values_from_oracle(oracle, observations)

    # 添加探索奖励
    visit_counts = get_visit_counts(observations)
    exploration_bonus = np.sqrt(np.log(total_visits) / (visit_counts + 1))

    ucb_q = raw_q + 0.1 * exploration_bonus
    return ucb_q
```

### 混合策略

```python
class HybridPlayer:
    """结合分类树和回归树"""

    def __init__(self, clf_tree, reg_tree):
        self.clf_tree = clf_tree
        self.reg_tree = reg_tree

    def predict(self, obs, player_id=1):
        # 先尝试分类树
        clf_action = self.clf_tree.predict(obs, player_id)

        legal_actions = np.where(obs == 0)[0]

        if clf_action in legal_actions:
            # 分类树合法，直接使用
            return clf_action
        else:
            # 分类树非法，使用回归树
            return self.reg_tree.predict(obs, player_id)
```

---

## 总结

### ✅ 推荐使用回归树的场景

1. **需要100%保证合法性的环境**
   - 棋类游戏（TicTacToe, Chess, Go）
   - 有硬约束的决策问题

2. **需要次优动作的场景**
   - 探索-利用权衡
   - 需要备选方案

3. **可解释性要求高的场景**
   - 需要知道"为什么选A而不是B"
   - 需要动作价值排序

### ❌ 不推荐的场景

1. **连续动作空间**
   - 回归树更适合离散动作

2. **极度关注模型大小**
   - MultiOutputRegressor会更大

3. **训练数据极少**
   - 需要为每个输出训练树

### 下一步

1. **实验对比**：在你的TicTacToe环境上运行对比实验
2. **调参优化**：调整max_depth、max_leaves等参数
3. **规则分析**：使用rule_extractor分析学到的Q值规则
4. **扩展到其他环境**：尝试在其他棋类游戏上应用

---

## 参考资料

- VIPER论文: "Verifiable Reinforcement Learning via Policy Extraction"
- sklearn文档: [DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
- sklearn文档: [MultiOutputRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html)
