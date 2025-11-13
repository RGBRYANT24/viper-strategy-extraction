# 规则提取功能更新总结

## 更新概述

为决策树规则提取系统添加了两个核心功能：
1. **优先级计算**：基于状态重要性对规则排序
2. **回归树支持**：完整保存9维logits向量，支持运行时mask

## 主要改动

### 1. Rule类扩展 ([model/rule_extractor.py](model/rule_extractor.py#L21-L117))

```python
class Rule:
    def __init__(self, antecedents, consequent, support_count=0,
                 priority=0.0, output_vector=None):
        self.output_vector = output_vector    # 新增：完整的9维向量
        self.priority = priority              # 新增：状态重要性

    def get_best_action(self, mask=None):     # 新增：考虑合法动作
        """根据mask选择最佳动作"""
```

**改进点：**
- 支持回归树的向量输出（9维logits）
- 添加优先级字段用于规则排序
- 提供`get_best_action()`方法自动处理不合法动作

### 2. DecisionTreeRuleExtractor增强 ([model/rule_extractor.py](model/rule_extractor.py#L141-L720))

#### 新增方法：

**优先级相关：**
- `compute_rule_state(rule)` - 找到规则对应的状态
- `compute_state_criticality(observation)` - 计算状态重要性
- `compute_rule_priorities(verbose)` - 为所有规则计算优先级
- `sort_rules_by_priority(descending)` - 按优先级排序

**回归树支持：**
- 自动检测树类型（分类/回归）
- 提取规则时保存完整的输出向量
- 导出时包含9维向量信息

**导出功能：**
- `export_rules_to_text(filepath, include_vectors)` - 导出文本格式
- `export_rules_to_json(filepath)` - 导出JSON格式（新增）

### 3. 命令行工具更新 ([extract_tree_rules.py](extract_tree_rules.py))

**新增参数：**
```bash
--compute-priority      # 计算规则优先级
--sort-by-priority      # 按优先级排序规则
```

**自动功能：**
- 自动检测回归树并保存完整向量
- 自动导出JSON格式（便于程序读取）

### 4. 文档和示例

**文档：**
- [RULE_PRIORITY_USAGE.md](RULE_PRIORITY_USAGE.md) - 完整使用说明

**示例脚本：**
- [example_use_regression_rules.py](example_use_regression_rules.py) - 演示如何使用规则
- [test_rule_priority.py](test_rule_priority.py) - 测试脚本

## 使用示例

### 提取规则（带优先级，推荐）⭐

```bash
python extract_tree_rules.py \
    --tree-path log/viper_mask_ppo_tree.joblib \
    --env-name TicTacToe-v0 \
    --oracle-path log/oracle_TicTacToe_ppo_aggressive.zip \
    --no-simplify \
    --compute-priority \
    --sort-by-priority \
    --output rules.txt
```

**重点**：优先级计算**不需要采样**（无需`--n-samples`），直接根据规则条件构造状态。

**输出：**
- `rules.txt` - 人类可读的文本格式
- `rules.json` - 机器可读的JSON格式

### 使用规则进行决策

```python
import json
import numpy as np

# 加载规则
with open('rules.json', 'r') as f:
    rules = json.load(f)['rules']

# 当前状态
observation = np.array([0, 1, 0, 0, -1, 0, 0, 0, 1])

# 找到匹配的规则
rule = find_matching_rule(observation, rules)

# 获取输出向量并应用mask
output_vector = np.array(rule['output_vector'])
mask = (observation == 0)  # 合法动作
logits = output_vector.copy()
logits[~mask] = -np.inf

# 选择最佳合法动作
action = np.argmax(logits)
```

## 核心优势

### 1. 回归树 vs 分类树

| 特性 | 分类树 | 回归树（本次更新） |
|------|--------|-------------------|
| 输出 | 单一类别 | 9维向量 |
| 合法性 | 需要额外处理 | 自动支持mask |
| 灵活性 | 低 | 高 |
| 推荐 | ❌ | ✅ |

### 2. 优先级排序

- **高优先级规则** = 关键决策点
- 自动识别最重要的规则
- 便于规则分析和压缩

### 3. 完整的工作流

```
训练回归树 → 提取规则 → 计算优先级 → 排序 → 导出 → 应用
    ↓           ↓          ↓         ↓      ↓      ↓
 viper_  extract_tree  compute_  sort_  .txt    使用
 mask_   rules.py      rule_     rules  .json   规则
 ppo.py               priorities                决策
```

## 输出格式对比

### 文本格式（.txt）

```
规则   1: IF X[4] <= 0.500 AND X[0] > 0.500
       THEN best_action = 4
       output_vector = [ 0.12 -0.34  0.56  0.23  0.89 -0.12  0.45  0.67 -0.23]
       (support=120, priority=2.3456)
```

### JSON格式（.json）

```json
{
  "antecedents": [[4, "<=", 0.5], [0, ">", 0.5]],
  "output_vector": [0.12, -0.34, 0.56, 0.23, 0.89, -0.12, 0.45, 0.67, -0.23],
  "best_action": 4,
  "support_count": 120,
  "priority": 2.3456
}
```

## 兼容性

### 向后兼容
- 仍然支持分类树（DecisionTreeClassifier）
- 不使用`--compute-priority`时不计算优先级
- 不使用`output_vector`字段时自动降级为分类模式

### Python版本
- 需要 Python 3.7+
- 依赖：numpy, sklearn, torch, gymnasium, sb3-contrib

## 测试方法

### 1. 运行基础测试
```bash
python test_rule_priority.py
```

### 2. 运行示例演示
```bash
python example_use_regression_rules.py --demo all
```

### 3. 使用真实模型测试
```bash
# 1. 提取规则
python extract_tree_rules.py \
    --tree-path log/your_tree.joblib \
    --env-name TicTacToe-v0 \
    --oracle-path log/your_oracle.zip \
    --compute-priority \
    --sort-by-priority

# 2. 使用规则
python example_use_regression_rules.py --rules rules.json
```

## 常见问题

### Q1: 规则文件很大怎么办？
A: 回归树规则包含9维向量，文件会比分类树大。可以：
1. 只保留高优先级规则
2. 压缩JSON文件
3. 使用二进制格式（如pickle）

### Q2: 如何只提取原始规则不简化？
A: 使用`--no-simplify`参数：
```bash
python extract_tree_rules.py --tree-path xxx.joblib --no-simplify
```

### Q3: 优先级计算很慢怎么办？
A: 优先级计算需要Oracle模型推理，可以：
1. 减少`--n-samples`参数
2. 先不计算优先级，只提取规则
3. 使用GPU加速Oracle推理

### Q4: 如何验证规则的准确性？
A: 使用规则在测试环境中运行，评估胜率：
```python
# 伪代码
for episode in range(1000):
    while not done:
        action = select_action(obs, rules)
        obs, reward, done = env.step(action)
    # 统计胜负
```

## 文件清单

### 核心文件
- `model/rule_extractor.py` - 规则提取核心逻辑（已修改）
- `extract_tree_rules.py` - 命令行工具（已修改）

### 文档
- `RULE_PRIORITY_USAGE.md` - 使用说明（新增）
- `SUMMARY_RULE_CHANGES.md` - 本文件（新增）

### 示例
- `example_use_regression_rules.py` - 使用示例（新增）
- `test_rule_priority.py` - 测试脚本（新增）

## 下一步建议

1. **训练回归树**：使用`train/viper_mask_ppo.py`训练
2. **提取规则**：使用`extract_tree_rules.py`提取并排序
3. **分析规则**：查看高优先级规则，理解策略
4. **应用规则**：在实际环境中使用规则进行决策
5. **评估性能**：对比规则策略与原始策略的表现
