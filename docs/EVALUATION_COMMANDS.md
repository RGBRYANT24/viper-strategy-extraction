# 决策树模型评估命令

本文档说明如何批量评估训练好的决策树模型。

## 快速开始

评估脚本位于 `test/evaluate_all_trees.py`，会自动识别模型的训练类型并使用正确的先后手设置：

- **X-only 模型**: 只作为先手测试 (`play_as_o_prob=0.0`)
- **XO-random 模型**: 随机先后手测试 (`play_as_o_prob=0.5`)

## 基本用法

```bash
# 在远程服务器上执行
cd /home/dai/PRIVE/Projects/Viper/viper-verifiable-rl-impl

# 激活虚拟环境
source activate viperenv  # 或 conda activate viperenv

# 使用默认设置：random 2000局，minmax 500局
python test/evaluate_all_trees.py log/viper_X-only
```

## 详细参数

```bash
python test/evaluate_all_trees.py MODEL_DIR [OPTIONS]

参数:
  MODEL_DIR              模型目录路径（必需）

选项:
  --opponents {random,minmax} [{random,minmax} ...]
                        对手类型（默认: random minmax）
  --n-random N          vs random 的测试局数（默认: 2000）
  --n-minmax N          vs minmax 的测试局数（默认: 500）
  --n-episodes N        所有对手的测试局数（会覆盖 --n-random 和 --n-minmax）
  -h, --help           显示帮助信息
```

## 示例命令

### 1. 默认设置（推荐）

```bash
# 对战 random 2000局 和 minmax 500局
# 这是最常用的设置：random多一些因为快，minmax少一些因为慢
python test/evaluate_all_trees.py log/viper_X-only
```

### 2. 自定义每个对手的局数

```bash
# random 3000局，minmax 300局
python test/evaluate_all_trees.py log/viper_X-only --n-random 3000 --n-minmax 300

# random 5000局，minmax 1000局
python test/evaluate_all_trees.py log/viper_X-only --n-random 5000 --n-minmax 1000
```

### 3. 所有对手使用相同局数

```bash
# 两种对手都是 1000 局
python test/evaluate_all_trees.py log/viper_X-only --n-episodes 1000
```

### 4. 只对战 random（多局数）

```bash
# 只测试 random，5000局
python test/evaluate_all_trees.py log/viper_X-only --opponents random --n-random 5000
```

### 5. 只对战 minmax

```bash
# 只测试 minmax，200局（minmax比较慢）
python test/evaluate_all_trees.py log/viper_X-only --opponents minmax --n-minmax 200
```

### 6. 快速测试（少量局数）

```bash
# 快速测试：random 100局，minmax 50局
python test/evaluate_all_trees.py log/viper_X-only --n-random 100 --n-minmax 50
```

## 推荐的完整评估流程

```bash
# 在远程服务器上执行

cd /home/dai/PRIVE/Projects/Viper/viper-verifiable-rl-impl
source activate viperenv

# 1. 使用推荐设置评估所有模型
# random 2000局（快速，需要更多样本），minmax 500局（慢，少量样本即可）
python test/evaluate_all_trees.py log/viper_X-only

# 2. 查看生成的报告文件
ls -lht log/viper_X-only/evaluation_*

# 报告文件说明:
# - evaluation_results_TIMESTAMP.json    # JSON 格式的详细结果
# - evaluation_report_TIMESTAMP.txt      # 人类可读的文本报告
```

## 输出说明

### 1. 控制台输出

脚本会实时显示：
- 每个模型的评估进度
- 每个对手的测试结果（胜/平/负比例）
- 汇总表格
- 最佳模型推荐

### 2. 保存的文件

评估完成后会在模型目录生成两个文件：

1. **JSON 结果** (`evaluation_results_TIMESTAMP.json`):
   - 完整的评估数据
   - 可用于进一步分析或可视化

2. **文本报告** (`evaluation_report_TIMESTAMP.txt`):
   - 人类可读的详细报告
   - 包含每个模型的完整评估结果

### 3. 结果解读

对于 **vs Random**:
- **优秀**: 胜率 > 90%
- **良好**: 胜率 70-90%
- **需改进**: 胜率 < 70%

对于 **vs MinMax**:
- **优秀**: 平局率 > 80%（说明学到了接近最优策略）
- **良好**: 平局率 60-80%
- **需改进**: 平局率 < 60%

## 注意事项

1. **先后手设置自动处理**: 脚本会根据模型文件名自动识别训练类型（X-only 或 XO-random）并使用正确的 play_as_o_prob 参数

2. **测试局数建议**:
   - 快速测试: random 100局，minmax 50局
   - 默认设置: random 2000局，minmax 500局（推荐）
   - 精确评估: random 5000局，minmax 1000局

3. **评估时间估计**:
   - 1000 局 vs random: 约 10-30 秒/模型
   - 1000 局 vs minmax: 约 30-60 秒/模型（因为 minmax 需要搜索）
   - 默认设置（random 2000 + minmax 500）：约 45-90 秒/模型

4. **为什么 random 多、minmax 少？**:
   - Random 对手每步都是随机的，需要更多局数才能得到稳定的统计结果
   - MinMax 对手是确定性的，相对少的局数就能准确评估性能
   - 同时 minmax 计算较慢，减少局数可以加快评估速度

5. **并发执行**: 如果有多个 GPU 或想加快速度，可以分别对 X-only 和 XO-random 模型运行评估

## 故障排查

### 问题: 找不到模型文件

```bash
# 检查目录内容
ls log/viper_X-only/*.joblib

# 确保路径正确
pwd
```

### 问题: 导入错误

```bash
# 确认在正确的虚拟环境
which python
python --version

# 确认必要的包已安装
pip list | grep gym
pip list | grep stable-baselines3
```

### 问题: 评估太慢

```bash
# 减少测试局数
python test/evaluate_all_trees.py log/viper_X-only --n-episodes 100

# 或只测试 random（比 minmax 快）
python test/evaluate_all_trees.py log/viper_X-only --opponents random
```

## 批量评估多个目录

如果需要评估多个目录，可以使用 shell 循环：

```bash
# 评估所有模型目录
for dir in log/viper_*; do
    if [ -d "$dir" ]; then
        echo "评估 $dir ..."
        python test/evaluate_all_trees.py "$dir" --n-episodes 1000
    fi
done
```

## 相关文件

- 评估脚本: [test/evaluate_all_trees.py](../test/evaluate_all_trees.py)
- VIPER 训练代码: [train/viper_mask_ppo.py](../train/viper_mask_ppo.py)
- Oracle 性能测试: [test/test_oracle_performance.py](../test/test_oracle_performance.py)
