# 规则与决策树一致性测试指南

## 概述

当从决策树导出规则文件后，需要确保规则文件和原始决策树在所有状态下的输出是一致的。本文档介绍如何进行一致性测试。

## 重要：先后手配置

在测试和评估时，**必须保持与训练时相同的先后手配置**：

- `play_as_o_prob=0.0`: 总是作为**先手(X)**
- `play_as_o_prob=0.5`: **随机先后手**（默认）
- `play_as_o_prob=1.0`: 总是作为**后手(O)**

### 如何确定训练时的配置

检查训练命令或训练脚本中的 `play_as_o_prob` 参数：

```bash
# 示例：只训练先手
python train/train_parallel_viper.py \
    --play-as-o-prob 0.0 \
    --log-prefix X-only
```

## 工具1：一致性检测 (`test_tree_rule_consistency.py`)

### 功能

检测决策树和导出规则在相同状态下是否产生相同的输出。

### 基本用法

```bash
# 测试1000个随机状态（默认随机先后手）
python test/test_tree_rule_consistency.py \
    --tree log/viper_X-only/viper_mask_ppo_tree_final.joblib \
    --rules log/viper_X-only/rules.json \
    --n-samples 1000
```

### 指定先后手配置

```bash
# 测试先手场景（与训练时配置一致）
python test/test_tree_rule_consistency.py \
    --tree log/viper_X-only/viper_mask_ppo_tree_final.joblib \
    --rules log/viper_X-only/rules.json \
    --n-samples 1000 \
    --play-as-o-prob 0.0

# 测试后手场景
python test/test_tree_rule_consistency.py \
    --tree log/viper_O-only/viper_mask_ppo_tree_final.joblib \
    --rules log/viper_O-only/rules.json \
    --n-samples 1000 \
    --play-as-o-prob 1.0
```

### 详细模式

```bash
# 显示所有不匹配的详细信息
python test/test_tree_rule_consistency.py \
    --tree log/viper_X-only/viper_mask_ppo_tree_final.joblib \
    --rules log/viper_X-only/rules.json \
    --n-samples 1000 \
    --play-as-o-prob 0.0 \
    --verbose \
    --max-mismatches 10
```

### 穷举测试（所有可能状态）

⚠️ **警告**：穷举测试会测试所有可能的井字棋状态，需要较长时间。

```bash
python test/test_tree_rule_consistency.py \
    --tree log/viper_X-only/viper_mask_ppo_tree_final.joblib \
    --rules log/viper_X-only/rules.json \
    --exhaustive \
    --play-as-o-prob 0.0
```

### 测试原始输出（不应用mask）

```bash
# 测试不应用合法动作mask的原始输出
python test/test_tree_rule_consistency.py \
    --tree log/viper_X-only/viper_mask_ppo_tree_final.joblib \
    --rules log/viper_X-only/rules.json \
    --n-samples 1000 \
    --play-as-o-prob 0.0 \
    --no-mask
```

### 输出解释

测试报告包含以下信息：

1. **总测试样本数**: 测试的状态数量
2. **无匹配规则**: 有多少状态无法匹配到任何规则（应该为0）
3. **向量数值不匹配**: 输出向量的数值有差异的状态数量
4. **Argmax不一致**: 虽然向量不同，但最优动作选择不同的状态数量（这是关键指标）
5. **数值精度**: 平均和最大的数值误差

#### 理想结果

```
✅ 所有测试通过！决策树和规则输出完全一致
```

#### 警告结果

```
⚠️  输出向量有细微差异，但动作选择一致
   这可能是浮点精度导致的，通常可以接受
```

#### 错误结果

```
❌ 发现动作选择不一致！
   可能原因：
   1. 规则导出时浮点数精度丢失
   2. 规则匹配逻辑有误
   3. Mask应用逻辑不同
```

## 工具2：规则策略评估 (`evaluate_rules.py`)

### 功能

评估规则策略的实际对战性能，并可与原始决策树对比。

### 基本用法

```bash
# 对战random对手（使用训练时的先后手配置）
python evaluation/evaluate_rules.py \
    --rules log/viper_X-only/rules.json \
    --opponent random \
    --n-episodes 100 \
    --play-as-o-prob 0.0
```

### 完整评估

```bash
# 评估所有对手类型
python evaluation/evaluate_rules.py \
    --rules log/viper_X-only/rules.json \
    --eval-all \
    --n-episodes 100 \
    --play-as-o-prob 0.0
```

### 与决策树对比

```bash
# 对比规则策略和决策树策略
python evaluation/evaluate_rules.py \
    --rules log/viper_X-only/rules.json \
    --tree-path log/viper_X-only/viper_mask_ppo_tree_final.joblib \
    --opponent minmax \
    --n-episodes 100 \
    --play-as-o-prob 0.0
```

### 只使用优先级最高的规则

```bash
# 只使用前20条优先级最高的规则
python evaluation/evaluate_rules.py \
    --rules log/viper_X-only/rules.json \
    --top-k 20 \
    --eval-all \
    --n-episodes 100 \
    --play-as-o-prob 0.0
```

## 完整测试流程

### 步骤1：导出规则

```bash
python analysis/export/export_rules.py \
    --tree-path log/viper_X-only/viper_mask_ppo_tree_final.joblib \
    --output rules.json \
    --compute-priority
```

### 步骤2：一致性检测

```bash
# 测试规则和决策树的一致性（使用与训练相同的先后手配置）
python test/test_tree_rule_consistency.py \
    --tree log/viper_X-only/viper_mask_ppo_tree_final.joblib \
    --rules log/viper_X-only/rules.json \
    --n-samples 1000 \
    --play-as-o-prob 0.0 \
    --verbose
```

### 步骤3：性能评估

```bash
# 评估规则策略性能并与决策树对比
python evaluation/evaluate_rules.py \
    --rules log/viper_X-only/rules.json \
    --tree-path log/viper_X-only/viper_mask_ppo_tree_final.joblib \
    --eval-all \
    --n-episodes 100 \
    --play-as-o-prob 0.0
```

### 步骤4：分析结果

如果一致性检测通过但性能有差异，可能原因：

1. **随机性**: 增加 `--n-episodes` 数量
2. **规则覆盖不完整**: 检查是否有 "无匹配规则" 的情况
3. **优先级排序**: 尝试使用 `--top-k` 参数测试不同数量的规则

## 常见问题

### Q1: 为什么一致性测试通过，但实际对战性能不同？

**A**: 可能原因：
- 测试的状态样本不够多，增加 `--n-samples`
- 某些罕见状态没有被测试到，考虑使用 `--exhaustive` 模式
- 对战时的随机性，增加 `--n-episodes`

### Q2: 如何修复精度丢失问题？

**A**: 在导出规则时使用更高精度：
```python
# 在 export_rules.py 中，使用 hex 格式存储浮点数
import json
json.dump(data, f, indent=2, default=lambda x: x.hex() if isinstance(x, float) else x)
```

### Q3: 训练时用的什么先后手配置？

**A**: 检查方法：
1. 查看训练命令中的 `--play-as-o-prob` 参数
2. 查看日志目录名称（如 `viper_X-only` 表示先手，`viper_O-only` 表示后手）
3. 如果未指定，默认值是 `0.5`（随机先后手）

### Q4: 先后手配置不匹配会怎样？

**A**: 会导致：
- 测试的状态分布与训练时不同
- 性能评估结果不准确
- 可能出现意外的高败率或高胜率

## 最佳实践

1. **始终明确指定** `--play-as-o-prob` 参数，不依赖默认值
2. **保持一致**: 测试和评估时使用与训练相同的先后手配置
3. **充分测试**: 对于关键模型，使用 `--exhaustive` 模式进行完整测试
4. **多次评估**: 对战评估至少使用 `--n-episodes 1000` 以减少随机性影响
5. **文档记录**: 在模型目录中记录训练时的 `play_as_o_prob` 配置

## 示例工作流

```bash
# 假设训练时使用 play_as_o_prob=0.0 (只训练先手)
TREE_PATH="log/viper_X-only/viper_mask_ppo_tree_final.joblib"
RULES_PATH="log/viper_X-only/rules.json"
PLAY_AS_O_PROB=0.0

# 1. 导出规则
python analysis/export/export_rules.py \
    --tree-path $TREE_PATH \
    --output $RULES_PATH \
    --compute-priority

# 2. 一致性检测（详细模式）
python test/test_tree_rule_consistency.py \
    --tree $TREE_PATH \
    --rules $RULES_PATH \
    --n-samples 5000 \
    --play-as-o-prob $PLAY_AS_O_PROB \
    --verbose \
    --max-mismatches 20

# 3. 性能评估（完整测试）
python evaluation/evaluate_rules.py \
    --rules $RULES_PATH \
    --tree-path $TREE_PATH \
    --eval-all \
    --n-episodes 1000 \
    --play-as-o-prob $PLAY_AS_O_PROB

# 4. 如果一致性通过，测试规则简化（top-k）
for k in 50 100 200 500; do
    echo "Testing top-$k rules..."
    python evaluation/evaluate_rules.py \
        --rules $RULES_PATH \
        --top-k $k \
        --eval-all \
        --n-episodes 200 \
        --play-as-o-prob $PLAY_AS_O_PROB
done
```

## 参考

- [规则提取文档](./RULE_EXTRACTION.md)
- [评估指南](./EVALUATION_COMMANDS.md)
- [先后手训练文档](./FIRST_SECOND_PLAYER_TRAINING.md)
