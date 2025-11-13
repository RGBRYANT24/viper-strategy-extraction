# 规则优先级功能使用说明

## 概述

本功能为决策树规则提取器添加了以下核心功能：
1. **优先级计算**：基于状态重要性（criticality）对规则排序
2. **回归树支持**：支持输出9维向量的回归树，完整保存logits用于排除不合法落子
3. **灵活导出**：支持文本和JSON两种格式

### 什么是状态重要性（Criticality）？

状态重要性衡量的是：在该状态下，选择不同动作对结果的影响有多大。

- **高重要性**：不同动作会导致显著不同的结果（关键决策点）
- **低重要性**：不同动作的结果差异不大（非关键状态）

计算公式：`criticality = max(log_prob) - min(log_prob)`

## 回归树 vs 分类树

### 分类树（Classification Tree）
- 输出：单一类别标签（int）
- 示例：`THEN class = 4`（直接预测动作4）

### 回归树（Regression Tree）⭐ 推荐
- 输出：9维向量（np.ndarray），对应井字棋9个位置的logits
- 示例：`THEN action = 4 (output_vector = [0.1, -0.2, ..., 0.8])`
- **优势**：可以在运行时根据合法动作mask选择最佳动作

## 功能实现

### 1. Rule 类扩展

为 `Rule` 类添加了 `priority` 和 `output_vector` 属性：

```python
class Rule:
    def __init__(self, antecedents, consequent, support_count=0,
                 priority=0.0, output_vector=None):
        self.priority = priority              # 优先级（状态重要性）
        self.output_vector = output_vector    # 完整的9维输出向量（回归树）

    def get_best_action(self, mask=None):
        """获取最佳动作（考虑合法动作mask）"""
        logits = self.output_vector.copy()
        if mask is not None:
            logits[~mask] = -np.inf  # 屏蔽不合法动作
        return np.argmax(logits)
```

### 2. DecisionTreeRuleExtractor 新增方法

#### `compute_rule_state(rule)`
将规则的前件转换为对应的棋盘状态（从训练数据中找到匹配该规则的样本）。

#### `compute_state_criticality(observation)`
计算给定状态的重要性分数，使用 Oracle 模型评估该状态下所有合法动作的 log 概率。

#### `compute_rule_priorities(verbose=False)`
为所有规则计算优先级：
1. 找到每条规则对应的状态
2. 计算该状态的重要性
3. 将重要性赋值给规则的 priority 属性

#### `sort_rules_by_priority(descending=True)`
按优先级对规则进行排序（默认降序，优先级高的在前）。

## 使用方法

### 命令行使用

#### 基础用法（只提取规则，不简化，不计算优先级）

```bash
python extract_tree_rules.py \
    --tree-path log/ppo_aggressive.joblib \
    --no-simplify
```

#### 计算优先级并排序（推荐，不需要采样）⭐

```bash
python extract_tree_rules.py \
    --tree-path log/ppo_aggressive.joblib \
    --env-name TicTacToe-v0 \
    --oracle-path log/oracle_TicTacToe_ppo_aggressive.zip \
    --no-simplify \
    --compute-priority \
    --sort-by-priority
```

**注意**：优先级计算不需要采样（`--n-samples`参数），直接根据规则条件构造状态喂给Oracle。

#### 完整示例（简化规则 + 计算优先级 + 排序）

```bash
python extract_tree_rules.py \
    --tree-path log/ppo_aggressive.joblib \
    --env-name TicTacToe-v0 \
    --oracle-path log/oracle_TicTacToe_ppo_aggressive.zip \
    --n-samples 5000 \
    --alpha 0.05 \
    --compute-priority \
    --sort-by-priority \
    --output rules_with_priority.txt
```

**说明**：
- 规则简化需要采样（`--n-samples 5000`）来做统计检验
- 优先级计算不需要采样，自动根据规则前件生成状态

### Python API 使用

```python
from model.tree_wrapper import TreeWrapper
from model.rule_extractor import DecisionTreeRuleExtractor
from sb3_contrib import MaskablePPO
import gymnasium as gym
import numpy as np

# 1. 加载决策树
tree_wrapper = TreeWrapper.load('log/ppo_aggressive.joblib')

# 2. 收集训练数据
env = gym.make('TicTacToe-v0', opponent_type='random')
oracle = MaskablePPO.load('log/oracle_TicTacToe_ppo_aggressive.zip', env=env)

# 采样数据
X_train, y_train = collect_training_data(oracle, env, n_samples=5000)

# 3. 创建规则提取器（传入 oracle 和 env）
extractor = DecisionTreeRuleExtractor(
    tree_wrapper.tree,
    X_train,
    y_train,
    oracle_model=oracle,
    env=env
)

# 4. 提取规则
extractor.extract_rules(verbose=True)

# 5. 计算优先级
extractor.compute_rule_priorities(verbose=True)

# 6. 按优先级排序
extractor.sort_rules_by_priority(descending=True)

# 7. 查看规则
extractor.print_rules(max_rules=10)

# 8. 查看统计信息
stats = extractor.get_stats()
print(f"优先级范围: [{stats['priority_min']:.4f}, {stats['priority_max']:.4f}]")
print(f"平均优先级: {stats['priority_mean']:.4f}")

# 9. 导出规则
extractor.export_rules_to_text('rules_with_priority.txt')
```

### 使用便捷函数

```python
from model.rule_extractor import extract_and_simplify_rules
from sb3_contrib import MaskablePPO
import gymnasium as gym

# 加载模型和环境
env = gym.make('TicTacToe-v0', opponent_type='random')
oracle = MaskablePPO.load('log/oracle_TicTacToe_ppo_aggressive.zip', env=env)

# 采样训练数据
X_train, y_train = collect_training_data(oracle, env, n_samples=5000)

# 一键提取、简化、计算优先级、排序
extractor = extract_and_simplify_rules(
    tree_model=tree,
    X_train=X_train,
    y_train=y_train,
    alpha=0.05,
    oracle_model=oracle,
    env=env,
    compute_priority=True,
    sort_by_priority=True,
    verbose=True
)

# 查看优先级最高的规则
extractor.print_rules(max_rules=10)
```

## 输出格式

### 文本格式 (.txt)

回归树规则会包含完整的9维向量：

```
决策树规则提取结果
================================================================================

树类型: 回归树 (Regression)

统计信息:
  n_rules: 50
  priority_min: 0.1234
  priority_max: 2.3456
  priority_mean: 1.2345

================================================================================

规则列表 (共 50 条):

规则   1: IF X[4] <= 0.500 AND X[0] > 0.500
       THEN best_action = 4
       output_vector = [ 0.12 -0.34  0.56  0.23  0.89 -0.12  0.45  0.67 -0.23]
       (support=120, priority=2.3456)

规则   2: IF X[4] > 0.500 AND X[8] <= 0.500
       THEN best_action = 8
       output_vector = [-0.45  0.23  0.12  0.67  0.34  0.89 -0.12  0.56  0.91]
       (support=85, priority=1.8923)
...
```

### JSON格式 (.json)

JSON格式便于程序读取和处理：

```json
{
  "tree_type": "regressor",
  "statistics": {
    "n_rules": 50,
    "priority_min": 0.1234,
    "priority_max": 2.3456,
    "priority_mean": 1.2345
  },
  "rules": [
    {
      "antecedents": [[4, "<=", 0.5], [0, ">", 0.5]],
      "output_vector": [0.12, -0.34, 0.56, 0.23, 0.89, -0.12, 0.45, 0.67, -0.23],
      "best_action": 4,
      "support_count": 120,
      "priority": 2.3456
    }
  ]
}
```

## 参数说明

### 命令行参数

- `--compute-priority`: 计算规则优先级（需要 `--env-name` 和 `--oracle-path`）
- `--sort-by-priority`: 按优先级对规则进行排序（需要 `--compute-priority`）
- `--env-name`: 环境名称（例如 `TicTacToe-v0`）
- `--oracle-path`: Oracle 模型路径（例如 `log/oracle_xxx.zip`）
- `--n-samples`: 采样数量（默认 5000）
- `--no-simplify`: 不进行规则简化（只提取原始规则）

### API 参数

- `oracle_model`: Oracle 模型对象（MaskablePPO）
- `env`: 环境对象（gymnasium.Env）
- `compute_priority`: 是否计算优先级（bool）
- `sort_by_priority`: 是否按优先级排序（bool）

## 工作流程

```
1. 加载决策树模型
   ↓
2. 提取决策树中的所有规则（IF-THEN 格式）
   ↓
3. [可选] 简化规则（删除统计学上不显著的条件）
   ↓
4. 为每条规则计算优先级
   ├─ 找到规则对应的棋盘状态（从训练数据中匹配）
   ├─ 使用 Oracle 模型计算该状态的重要性
   └─ 将重要性赋值给规则的 priority 属性
   ↓
5. [可选] 按优先级排序规则
   ↓
6. 输出和导出规则
```

## 注意事项

1. **需要 Oracle 模型**：计算优先级需要提供训练好的 Oracle 模型（MaskablePPO）
2. **需要环境对象**：需要 gymnasium 环境对象来支持状态评估
3. **采样数量**：
   - **规则简化**需要采样：建议使用至少 5000 个样本（`--n-samples 5000`）
   - **优先级计算**不需要采样：自动根据规则前件生成状态
4. **计算时间**：优先级计算会增加运行时间，因为需要为每条规则计算状态重要性
5. **排序**：如果不使用 `--sort-by-priority`，规则将保持原始顺序（按决策树遍历顺序）
6. **分离使用**：可以单独使用优先级计算（加`--no-simplify`）或单独使用规则简化（不加`--compute-priority`）

## 使用规则进行决策（回归树）

### 加载规则并应用mask

```python
import json
import numpy as np

# 1. 加载规则
with open('rules.json', 'r') as f:
    data = json.load(f)
    rules = data['rules']

# 2. 找到匹配当前状态的规则
def find_matching_rule(observation, rules):
    """找到匹配当前状态的规则"""
    for rule in rules:
        # 检查所有前件条件
        match = True
        for feature_idx, operator, value in rule['antecedents']:
            if operator == '<=':
                if not (observation[feature_idx] <= value):
                    match = False
                    break
            else:  # operator == '>'
                if not (observation[feature_idx] > value):
                    match = False
                    break
        if match:
            return rule
    return None

# 3. 选择动作（考虑合法动作mask）
def select_action(observation, rules):
    """
    根据规则选择动作，自动排除不合法落子

    Args:
        observation: 当前棋盘状态 (9维向量)
        rules: 规则列表

    Returns:
        最佳合法动作
    """
    # 找到匹配的规则
    rule = find_matching_rule(observation, rules)
    if rule is None:
        return None

    # 获取输出向量
    output_vector = np.array(rule['output_vector'])

    # 计算合法动作mask（井字棋：空位置为合法）
    mask = (observation == 0)

    # 应用mask：将不合法动作设为-inf
    logits = output_vector.copy()
    logits[~mask] = -np.inf

    # 选择最佳合法动作
    best_action = np.argmax(logits)

    return best_action

# 4. 使用示例
observation = np.array([0, 1, 0, 0, -1, 0, 0, 0, 1])  # 井字棋状态
action = select_action(observation, rules)
print(f"选择动作: {action}")

# 5. 查看规则详情
rule = find_matching_rule(observation, rules)
print(f"匹配规则的输出向量: {rule['output_vector']}")
print(f"优先级: {rule['priority']:.4f}")
```

## 应用场景

1. **在线决策**：使用规则集进行实时决策，自动处理不合法动作
2. **策略理解**：分析高优先级规则，了解策略的关键决策点
3. **规则验证**：验证规则在不同对手下的表现
4. **规则压缩**：只保留高优先级规则，减少规则数量
5. **可解释性**：向用户展示最重要的决策规则和完整的logits向量

## 示例输出

```bash
================================================================================
步骤 4: 计算规则优先级
================================================================================

计算 50 条规则的优先级...
  规则 1: priority=2.3456
  规则 2: priority=1.8923
  规则 3: priority=1.5678
  ...
完成优先级计算
  优先级范围: [0.1234, 2.3456]
  平均优先级: 1.2345

按优先级排序规则（降序）...

优先级最高的10条规则:
================================================================================
规则   1: IF X[4] <= 0.500 AND X[0] > 0.500 THEN class = 4 (support=120, priority=2.3456)
规则   2: IF X[4] > 0.500 AND X[8] <= 0.500 THEN class = 8 (support=85, priority=1.8923)
...
================================================================================

完成！
规则已保存到: rules_with_priority.txt
总共提取 50 条规则
优先级范围: [0.1234, 2.3456]
平均优先级: 1.2345
规则已按优先级降序排序
```
