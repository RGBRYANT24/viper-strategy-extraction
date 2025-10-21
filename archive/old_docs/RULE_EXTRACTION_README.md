# 决策树规则提取器使用指南

本指南介绍如何使用新添加的决策树规则提取和简化功能。

## 📋 目录

- [功能概述](#功能概述)
- [新增文件](#新增文件)
- [安装依赖](#安装依赖)
- [快速开始](#快速开始)
- [服务器部署命令](#服务器部署命令)
- [详细使用说明](#详细使用说明)
- [API文档](#api文档)

---

## 功能概述

规则提取器可以将训练好的sklearn决策树转换为易于理解的IF-THEN规则集合，并使用统计学方法简化这些规则。

### 主要特性

1. **规则提取**: 从决策树的每个叶子节点提取IF-THEN规则
2. **统计简化**: 使用卡方检验、Yates校正和Fisher精确检验删除不必要的条件
3. **规则分析**: 识别默认规则和规则分布
4. **无缝集成**: 已集成到项目的TreeWrapper类中

### 统计方法

- **卡方检验** (χ²): 当期望频数 ≥ 10 时使用
- **Yates连续性校正**: 当期望频数 ≥ 5 时使用
- **Fisher精确检验**: 当期望频数 < 5 时使用

---

## 新增文件

```
viper-verifiable-rl-impl/
├── model/
│   └── rule_extractor.py          # 规则提取和简化核心模块
├── test/
│   └── test_rule_extractor.py     # 单元测试（25个测试用例）
├── extract_tree_rules.py          # 命令行工具：从VIPER训练的树提取规则
├── demo_rule_extraction.py        # 演示脚本：使用Iris数据集演示
└── RULE_EXTRACTION_README.md      # 本文档
```

### 修改的文件

- `model/tree_wrapper.py`: 添加了规则提取相关方法
- `requirements.txt`: 添加了scipy依赖

---

## 安装依赖

### 在服务器上执行

```bash
# 进入项目目录
cd /path/to/viper-verifiable-rl-impl

# 安装scipy（新增依赖）
pip install scipy==1.11.4

# 或者重新安装所有依赖
pip install -r requirements.txt
```

---

## 快速开始

### 1. 运行演示脚本（推荐）

这个脚本使用Iris数据集演示完整的规则提取流程：

```bash
python demo_rule_extraction.py
```

**输出示例**:
```
================================================================================
决策树规则提取器演示
================================================================================

步骤 1: 加载Iris数据集
--------------------------------------------------------------------------------
训练集大小: 105
测试集大小: 45
特征数量: 4
...

规则 1:
  IF X[2] <= 2.450 THEN class = 0 (support=37)
  类别: setosa

规则 2:
  IF X[2] > 2.450 AND X[3] <= 1.750 AND X[2] <= 4.950 THEN class = 1 (support=36)
  类别: versicolor
...
```

### 2. 从VIPER训练的树提取规则

#### 基础用法（不进行统计简化）

```bash
python extract_tree_rules.py \
  --tree-path log/viper_TicTacToe-v0_all-leaves_50.joblib
```

#### 完整用法（包含规则简化）

```bash
python extract_tree_rules.py \
  --tree-path log/viper_TicTacToe-v0_all-leaves_50.joblib \
  --env-name TicTacToe-v0 \
  --oracle-path log/oracle_TicTacToe_selfplay.zip \
  --n-samples 5000 \
  --alpha 0.05 \
  --output my_rules.txt
```

**参数说明**:
- `--tree-path`: 决策树模型文件路径（必需）
- `--env-name`: 环境名称（用于收集训练数据）
- `--oracle-path`: Oracle模型路径（用于生成样本）
- `--n-samples`: 用于规则分析的样本数（默认5000）
- `--alpha`: 显著性水平（默认0.05）
- `--output`: 输出文件路径（默认自动生成）

---

## 服务器部署命令

### 1. 上传代码到服务器

```bash
# 在本地执行
scp model/rule_extractor.py user@server:/path/to/viper-verifiable-rl-impl/model/
scp model/tree_wrapper.py user@server:/path/to/viper-verifiable-rl-impl/model/
scp test/test_rule_extractor.py user@server:/path/to/viper-verifiable-rl-impl/test/
scp extract_tree_rules.py user@server:/path/to/viper-verifiable-rl-impl/
scp demo_rule_extraction.py user@server:/path/to/viper-verifiable-rl-impl/
scp requirements.txt user@server:/path/to/viper-verifiable-rl-impl/
```

或者使用git推送：
```bash
git add model/rule_extractor.py model/tree_wrapper.py test/test_rule_extractor.py \
        extract_tree_rules.py demo_rule_extraction.py requirements.txt RULE_EXTRACTION_README.md

git commit -m "添加决策树规则提取和简化功能

- 添加rule_extractor.py模块，支持规则提取和统计简化
- 集成到TreeWrapper类
- 添加完整的单元测试（25个测试用例）
- 提供命令行工具和演示脚本"

git push origin transfer_dt
```

### 2. 在服务器上安装依赖

```bash
# SSH登录服务器
ssh user@server

# 进入项目目录
cd /path/to/viper-verifiable-rl-impl

# 拉取最新代码（如果使用git）
git pull origin transfer_dt

# 安装scipy
pip install scipy==1.11.4
```

### 3. 运行测试

```bash
# 运行单元测试
python -m pytest test/test_rule_extractor.py -v

# 或者直接运行测试文件
python test/test_rule_extractor.py

# 运行演示
python demo_rule_extraction.py
```

**期望的测试输出**:
```
test_rule_extractor.py::TestRule::test_rule_creation PASSED
test_rule_extractor.py::TestRule::test_rule_str PASSED
test_rule_extractor.py::TestRule::test_rule_matches PASSED
...
========================= 25 passed in 2.34s =========================
```

### 4. 从现有决策树提取规则

假设你已经训练了决策树模型：

```bash
# 查看现有的决策树模型
ls -lh log/*.joblib

# 示例输出：
# log/viper_TicTacToe-v0_all-leaves_50.joblib
# log/viper_TicTacToe-v0_minmax-leaves_5.joblib

# 提取规则（基础版本）
python extract_tree_rules.py \
  --tree-path log/viper_TicTacToe-v0_all-leaves_50.joblib

# 提取并简化规则（推荐）
python extract_tree_rules.py \
  --tree-path log/viper_TicTacToe-v0_all-leaves_50.joblib \
  --env-name TicTacToe-v0 \
  --oracle-path log/oracle_TicTacToe_selfplay.zip \
  --n-samples 5000 \
  --alpha 0.05

# 查看生成的规则文件
cat log/viper_TicTacToe-v0_all-leaves_50.rules.txt
```

---

## 详细使用说明

### 在Python代码中使用

#### 方法1: 使用TreeWrapper（推荐）

```python
from model.tree_wrapper import TreeWrapper
import numpy as np

# 1. 加载训练好的决策树
tree_wrapper = TreeWrapper.load('log/viper_TicTacToe-v0_all-leaves_50.joblib')

# 2. 设置训练数据（从VIPER训练过程中收集）
X_train = np.array([...])  # 你的训练数据特征
y_train = np.array([...])  # 你的训练数据标签
tree_wrapper.set_training_data(X_train, y_train)

# 3. 提取并简化规则
extractor = tree_wrapper.extract_rules(alpha=0.05, verbose=True)

# 4. 查看规则
tree_wrapper.print_rules(max_rules=10)

# 5. 导出规则到文件
tree_wrapper.export_rules('output_rules.txt')

# 6. 获取规则列表
rules = tree_wrapper.get_rules()
for i, rule in enumerate(rules):
    print(f"规则 {i+1}: {rule}")
```

#### 方法2: 直接使用DecisionTreeRuleExtractor

```python
from sklearn.tree import DecisionTreeClassifier
from model.rule_extractor import DecisionTreeRuleExtractor
import numpy as np

# 假设你已经有了训练好的决策树
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)

# 创建规则提取器
extractor = DecisionTreeRuleExtractor(
    tree_model=clf,
    X_train=X_train,
    y_train=y_train,
    feature_names=['feature_0', 'feature_1', ...],  # 可选
    alpha=0.05  # 显著性水平
)

# 步骤1: 提取规则
extractor.extract_rules(verbose=True)
print(f"提取了 {len(extractor.rules)} 条规则")

# 步骤2: 简化规则
extractor.simplify_rules(verbose=True)

# 步骤3: 分析规则分布
rules, default_consequent = extractor.eliminate_redundant_rules(verbose=True)

# 查看规则
extractor.print_rules()

# 导出规则
extractor.export_rules_to_text('rules_output.txt')
```

#### 方法3: 使用便捷函数

```python
from model.rule_extractor import extract_and_simplify_rules

# 一行代码完成提取和简化
extractor = extract_and_simplify_rules(
    tree_model=clf,
    X_train=X_train,
    y_train=y_train,
    feature_names=None,  # 可选
    alpha=0.05,
    verbose=True
)

# 获取规则
rules = extractor.rules
```

---

## API文档

### Rule类

表示一条IF-THEN规则。

```python
class Rule:
    def __init__(self, antecedents, consequent, support_count=0):
        """
        Args:
            antecedents: List[(feature_idx, operator, value)]
                        前件列表，operator in ['<=', '>']
            consequent: int - 预测的类别/动作
            support_count: int - 支持该规则的样本数量
        """
```

**主要方法**:
- `matches(X)`: 判断样本X是否匹配该规则
- `to_dict()`: 转换为字典格式
- `__str__()`: 返回可读的规则字符串

### DecisionTreeRuleExtractor类

从决策树中提取和简化规则的主类。

```python
class DecisionTreeRuleExtractor:
    def __init__(self, tree_model, X_train, y_train,
                 feature_names=None, alpha=0.05):
        """
        Args:
            tree_model: sklearn.tree.DecisionTreeClassifier
            X_train: np.ndarray - 训练数据特征
            y_train: np.ndarray - 训练数据标签
            feature_names: List[str] - 特征名称（可选）
            alpha: float - 显著性水平，默认0.05
        """
```

**主要方法**:

1. `extract_rules(verbose=False)` → `List[Rule]`
   - 从决策树中提取所有规则
   - 每个叶子节点生成一条规则

2. `simplify_rules(verbose=False)` → `List[Rule]`
   - 使用统计检验简化规则
   - 删除与结论独立的前件

3. `eliminate_redundant_rules(verbose=False)` → `(List[Rule], int)`
   - 分析规则分布
   - 识别默认规则（最常见的结论）

4. `build_contingency_table(rule, antecedent_idx)` → `np.ndarray`
   - 为规则的某个前件构建2x2列联表

5. `test_independence(contingency_table)` → `(bool, float, float)`
   - 进行独立性检验
   - 返回 (is_independent, chi2_stat, p_value)

6. `print_rules(max_rules=None)`
   - 打印规则到控制台

7. `export_rules_to_text(filepath)`
   - 导出规则到文本文件

8. `get_stats()` → `Dict`
   - 获取提取统计信息

### TreeWrapper扩展方法

TreeWrapper类新增的规则提取相关方法：

```python
# 设置训练数据
tree_wrapper.set_training_data(X_train, y_train)

# 提取规则
extractor = tree_wrapper.extract_rules(alpha=0.05, verbose=True)

# 获取规则
rules = tree_wrapper.get_rules()

# 打印规则
tree_wrapper.print_rules(max_rules=10)

# 导出规则
tree_wrapper.export_rules('output.txt')
```

---

## 使用场景

### 场景1: 分析VIPER训练的决策树策略

```bash
# 1. 训练VIPER决策树
python train_viper_improved.py \
  --env-name TicTacToe-v0 \
  --oracle-path log/oracle_TicTacToe_selfplay.zip \
  --n-iter 80 \
  --max-leaves 50

# 2. 提取并分析规则
python extract_tree_rules.py \
  --tree-path log/viper_TicTacToe-v0_all-leaves_50.joblib \
  --env-name TicTacToe-v0 \
  --oracle-path log/oracle_TicTacToe_selfplay.zip \
  --n-samples 5000

# 3. 查看规则
cat log/viper_TicTacToe-v0_all-leaves_50.rules.txt
```

### 场景2: 对比简化前后的规则

```python
from model.rule_extractor import DecisionTreeRuleExtractor

# 创建提取器
extractor = DecisionTreeRuleExtractor(clf, X_train, y_train)

# 提取原始规则
extractor.extract_rules(verbose=True)
print(f"\n原始规则（{len(extractor.rules)}条）:")
extractor.print_rules()

# 简化规则
extractor.simplify_rules(verbose=True)
print(f"\n简化后规则（{len(extractor.rules)}条）:")
extractor.print_rules()

# 查看统计信息
stats = extractor.get_stats()
print(f"删除了 {stats['n_removed_antecedents']} 个前件")
```

### 场景3: 将规则集成到验证系统

```python
# 加载决策树和规则
tree_wrapper = TreeWrapper.load('model.joblib')
tree_wrapper.set_training_data(X_train, y_train)
extractor = tree_wrapper.extract_rules(verbose=False)

# 获取规则
rules = extractor.rules

# 验证某个状态
state = env.reset()
matching_rules = [r for r in rules if r.matches(state)]

print(f"状态 {state} 匹配 {len(matching_rules)} 条规则:")
for rule in matching_rules:
    print(f"  {rule}")
    print(f"  建议动作: {rule.consequent}")
```

---

## 常见问题

### Q1: 规则简化会改变决策树的预测结果吗？

**答**: 不会。规则简化只是删除统计学上不显著的条件，使规则更简洁，但不改变预测结果。提取的规则与原决策树的预测完全一致。

### Q2: 什么时候应该使用规则简化？

**答**: 当你需要更简洁、更易理解的规则时。简化后的规则保留了关键的决策条件，删除了冗余的条件。

### Q3: alpha参数如何选择？

**答**:
- `alpha=0.05`（默认）: 标准显著性水平，平衡简化程度和准确性
- `alpha=0.01`: 更严格，简化程度较小，保留更多条件
- `alpha=0.10`: 更宽松，简化程度较大，规则更简洁

### Q4: 需要多少训练样本进行规则简化？

**答**: 建议至少1000个样本，5000个样本更佳。样本越多，统计检验越可靠。

### Q5: 如何处理大型决策树？

**答**:
```python
# 只打印前20条规则
extractor.print_rules(max_rules=20)

# 或者筛选支持度高的规则
important_rules = [r for r in extractor.rules if r.support_count > 10]
```

---

## 技术细节

### 统计检验流程

```
对每条规则的每个前件:
1. 构建2x2列联表
   ┌───────────────┬──────────┬────────────┐
   │               │ 符合结论 │ 不符合结论 │
   ├───────────────┼──────────┼────────────┤
   │ 符合该前件    │   x11    │    x12     │
   │ 不符合该前件  │   x21    │    x22     │
   └───────────────┴──────────┴────────────┘

2. 选择检验方法:
   - max(期望频数) >= 10 → 卡方检验
   - max(期望频数) >= 5  → Yates校正
   - max(期望频数) < 5   → Fisher精确检验

3. 计算p值，判断是否独立:
   - p > α → 独立 → 删除该前件
   - p ≤ α → 不独立 → 保留该前件
```

### 复杂度分析

- **规则提取**: O(n) - n为叶子节点数
- **规则简化**: O(r × c × m) - r为规则数，c为平均前件数，m为训练样本数
- **内存占用**: O(r × c + m) - 主要由规则和训练数据决定

---

## 更新日志

### v1.0 (2025-10-15)

- ✅ 实现了Rule类，表示IF-THEN规则
- ✅ 实现了DecisionTreeRuleExtractor类，支持规则提取和简化
- ✅ 集成到TreeWrapper，无缝使用
- ✅ 添加了25个单元测试，覆盖所有主要功能
- ✅ 提供了命令行工具extract_tree_rules.py
- ✅ 提供了演示脚本demo_rule_extraction.py
- ✅ 支持三种统计检验方法（卡方、Yates、Fisher）
- ✅ 添加scipy依赖

---

## 贡献者

- 原始规则提取算法: 用户提供
- 项目集成和测试: VIPER项目组

---

## 许可证

与主项目保持一致。

---

## 反馈和支持

如有问题或建议，请提交Issue或联系项目维护者。
