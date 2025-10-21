# 规则提取快速入门 🚀

## 服务器部署三步走

### 步骤1: 安装依赖
```bash
pip install scipy==1.11.4
```

### 步骤2: 运行测试
```bash
# 运行完整测试套件
python -m pytest test/test_rule_extractor.py -v

# 或者直接运行
python test/test_rule_extractor.py

# 运行演示
python demo_rule_extraction.py
```

### 步骤3: 提取规则
```bash
# 基础用法
python extract_tree_rules.py --tree-path log/viper_TicTacToe-v0_all-leaves_50.joblib

# 完整用法（推荐）
python extract_tree_rules.py \
  --tree-path log/viper_TicTacToe-v0_all-leaves_50.joblib \
  --env-name TicTacToe-v0 \
  --oracle-path log/oracle_TicTacToe_selfplay.zip \
  --n-samples 5000 \
  --alpha 0.05
```

---

## 常用命令

### 1. 查看可用的决策树模型
```bash
ls -lh log/*.joblib
```

### 2. 提取规则（不简化）
```bash
python extract_tree_rules.py --tree-path <你的模型路径>
```

### 3. 提取并简化规则（推荐）
```bash
python extract_tree_rules.py \
  --tree-path <你的模型路径> \
  --env-name TicTacToe-v0 \
  --oracle-path <你的oracle路径> \
  --n-samples 5000
```

### 4. 查看规则输出
```bash
cat <模型路径>.rules.txt
```

---

## Python代码示例

### 快速使用
```python
from model.tree_wrapper import TreeWrapper
import numpy as np

# 加载树
tree = TreeWrapper.load('log/viper_TicTacToe-v0_all-leaves_50.joblib')

# 设置训练数据
tree.set_training_data(X_train, y_train)

# 提取规则
extractor = tree.extract_rules(alpha=0.05, verbose=True)

# 打印规则
tree.print_rules(max_rules=10)

# 导出规则
tree.export_rules('my_rules.txt')
```

---

## 预期输出示例

```
================================================================================
步骤 1: 从决策树提取规则
================================================================================
提取到 15 条规则
平均每条规则有 3.2 个前件

原始规则:
规则   1: IF X[0] <= 5.500 AND X[1] > 3.000 THEN class = 0 (support=42)
规则   2: IF X[0] > 5.500 AND X[2] <= 4.950 THEN class = 1 (support=38)
...

================================================================================
步骤 2: 简化规则（删除不必要的前件）
================================================================================
  规则 3, 删除前件 1: (1, '>', 2.800) (χ²=0.234, p=0.628)
  规则 7, 删除前件 0: (0, '<=', 6.200) (χ²=1.152, p=0.283)

简化完成: 删除了 5 个前件
简化后保留 15 条规则

================================================================================
步骤 3: 分析规则分布
================================================================================
最常见的结论: 1 (出现 7 次，占 46.7%)
所有结论分布: {0: 5, 1: 7, 2: 3}
```

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `model/rule_extractor.py` | 核心模块：Rule类和DecisionTreeRuleExtractor类 |
| `model/tree_wrapper.py` | 已集成规则提取方法 |
| `test/test_rule_extractor.py` | 25个单元测试 |
| `extract_tree_rules.py` | 命令行工具 |
| `demo_rule_extraction.py` | Iris数据集演示 |
| `RULE_EXTRACTION_README.md` | 完整文档 |

---

## 参数速查

### extract_tree_rules.py 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--tree-path` | 决策树路径（必需） | - |
| `--env-name` | 环境名称 | None |
| `--oracle-path` | Oracle路径 | None |
| `--n-samples` | 采样数量 | 5000 |
| `--alpha` | 显著性水平 | 0.05 |
| `--output` | 输出文件 | 自动生成 |
| `--no-simplify` | 不进行简化 | False |
| `--max-rules` | 打印最大规则数 | None（全部） |

### alpha值选择

- `0.01`: 严格，保留更多条件
- `0.05`: 标准（推荐）
- `0.10`: 宽松，规则更简洁

---

## 故障排查

### 问题1: ModuleNotFoundError: No module named 'scipy'
```bash
pip install scipy==1.11.4
```

### 问题2: 找不到决策树文件
```bash
# 检查文件是否存在
ls -lh log/*.joblib

# 使用正确的路径
python extract_tree_rules.py --tree-path $(ls log/*.joblib | head -1)
```

### 问题3: 提取规则但未简化
需要同时提供 `--env-name` 和 `--oracle-path` 参数。

---

## 下一步

- 📖 阅读完整文档: [RULE_EXTRACTION_README.md](RULE_EXTRACTION_README.md)
- 🧪 运行测试: `python test/test_rule_extractor.py`
- 🎯 运行演示: `python demo_rule_extraction.py`
- 🚀 提取你的规则: `python extract_tree_rules.py --tree-path <your-model>`

---

**需要帮助？** 查看 `RULE_EXTRACTION_README.md` 获取详细文档和API参考。
