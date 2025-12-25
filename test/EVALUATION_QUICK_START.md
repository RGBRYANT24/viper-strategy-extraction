# 决策树评估 - 快速开始

## 最简单的用法

在远程服务器上运行：

```bash
cd /home/dai/PRIVE/Projects/Viper/viper-verifiable-rl-impl
source activate viperenv

# 使用默认设置（random 2000局，minmax 500局）
python test/evaluate_all_trees.py log/viper_X-only
```

就这么简单！脚本会：
- ✓ 自动识别 X-only 和 XO-random 模型
- ✓ 对 X-only 使用 play_as_o_prob=0.0（只先手）
- ✓ 对 XO-random 使用 play_as_o_prob=0.5（随机先后手）
- ✓ 测试 vs random（2000局）和 vs minmax（500局）
- ✓ 生成汇总表格和详细报告

## 常用命令

```bash
# 1. 默认设置（推荐）
python test/evaluate_all_trees.py log/viper_X-only

# 2. 自定义局数
python test/evaluate_all_trees.py log/viper_X-only --n-random 3000 --n-minmax 300

# 3. 快速测试
python test/evaluate_all_trees.py log/viper_X-only --n-random 100 --n-minmax 50

# 4. 只测试 random
python test/evaluate_all_trees.py log/viper_X-only --opponents random --n-random 5000

# 5. 只测试 minmax
python test/evaluate_all_trees.py log/viper_X-only --opponents minmax --n-minmax 200
```

## 输出文件

运行后会在模型目录生成：
- `evaluation_results_TIMESTAMP.json` - JSON格式详细结果
- `evaluation_report_TIMESTAMP.txt` - 人类可读的文本报告

## 参数说明

```
--n-random N     vs random 的测试局数（默认: 2000）
--n-minmax N     vs minmax 的测试局数（默认: 500）
--n-episodes N   所有对手使用相同局数（会覆盖上面两个参数）
--opponents      选择对手类型（默认: random minmax）
```

## 为什么 random 多、minmax 少？

- **Random**: 随机对手，需要更多局数获得稳定统计（建议 2000+）
- **MinMax**: 确定性对手，少量局数即可准确评估（建议 500-1000）
- **速度**: minmax 计算慢，减少局数可加快评估

## 详细文档

查看完整文档：[docs/EVALUATION_COMMANDS.md](../docs/EVALUATION_COMMANDS.md)
