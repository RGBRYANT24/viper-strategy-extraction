# 并行训练所有先后手配置 - 命令集合

## 配置说明

- **深度范围**: 5-25 (步长 5) → [5, 10, 15, 20, 25] = 5个配置
- **叶子节点范围**: 25-150 (步长 25) → [25, 50, 75, 100, 125, 150] = 6个配置
- **先后手配置**: 3种
  - X-only: `--play-as-o-prob 0.0` (只训练先手)
  - XO-random: `--play-as-o-prob 0.5` (先后手随机)
  - O-only: `--play-as-o-prob 1.0` (只训练后手)
- **总配置数**: 5 × 6 × 3 = **90 个决策树**

## 推荐参数

```bash
ORACLE_PATH="log/oracle_TicTacToe_ppo_aggressive.zip"
N_ITERATIONS=15
SAMPLES_PER_ITER=10000
MIN_SAMPLES_SPLIT=5
MIN_SAMPLES_LEAF=2
```

---

## 命令 1: 训练只先手配置 (X-only)

```bash
python train/train_parallel_viper.py \
  --oracle-path log/oracle_TicTacToe_ppo_aggressive.zip \
  --depth-range 5 25 \
  --depth-step 5 \
  --leaves-range 25 150 \
  --leaves-step 25 \
  --n-iterations 15 \
  --samples-per-iter 10000 \
  --play-as-o-prob 0.0 \
  --min-samples-split 5 \
  --min-samples-leaf 2 \
  --output-dir log/viper_X-only \
  --yes
```

**生成文件**: 30个模型 (5个深度 × 6个叶子节点)
- 文件名示例: `viper_mask_ppo_tree_..._X-only_iter15_samples10000_depth5_leaves25.joblib`
- 输出目录示例: `log/viper_X-only_20250117_143022_d5-25_l25-150_iter15/`

---

## 命令 2: 训练先后手随机配置 (XO-random)

```bash
python train/train_parallel_viper.py \
  --oracle-path log/oracle_TicTacToe_ppo_aggressive.zip \
  --depth-range 5 25 \
  --depth-step 5 \
  --leaves-range 25 150 \
  --leaves-step 25 \
  --n-iterations 15 \
  --samples-per-iter 10000 \
  --play-as-o-prob 0.5 \
  --min-samples-split 5 \
  --min-samples-leaf 2 \
  --output-dir log/viper_XO-random \
  --yes
```

**生成文件**: 30个模型
- 文件名示例: `viper_mask_ppo_tree_..._XO-random_iter15_samples10000_depth5_leaves25.joblib`
- 输出目录示例: `log/viper_XO-random_20250117_143500_d5-25_l25-150_iter15/`

---

## 命令 3: 训练只后手配置 (O-only)

```bash
python train/train_parallel_viper.py \
  --oracle-path log/oracle_TicTacToe_ppo_aggressive.zip \
  --depth-range 5 25 \
  --depth-step 5 \
  --leaves-range 25 150 \
  --leaves-step 25 \
  --n-iterations 15 \
  --samples-per-iter 10000 \
  --play-as-o-prob 1.0 \
  --min-samples-split 5 \
  --min-samples-leaf 2 \
  --output-dir log/viper_O-only \
  --yes
```

**生成文件**: 30个模型
- 文件名示例: `viper_mask_ppo_tree_..._O-only_iter15_samples10000_depth5_leaves25.joblib`
- 输出目录示例: `log/viper_O-only_20250117_144000_d5-25_l25-150_iter15/`

---

## 一次性运行所有命令（顺序执行）

```bash
# 1. X-only
python train/train_parallel_viper.py \
  --oracle-path log/oracle_TicTacToe_ppo_aggressive.zip \
  --depth-range 5 25 --depth-step 5 \
  --leaves-range 25 150 --leaves-step 25 \
  --n-iterations 15 --samples-per-iter 10000 \
  --play-as-o-prob 0.0 \
  --min-samples-split 5 --min-samples-leaf 2 \
  --output-dir log/viper_X-only --yes

# 2. XO-random
python train/train_parallel_viper.py \
  --oracle-path log/oracle_TicTacToe_ppo_aggressive.zip \
  --depth-range 5 25 --depth-step 5 \
  --leaves-range 25 150 --leaves-step 25 \
  --n-iterations 15 --samples-per-iter 10000 \
  --play-as-o-prob 0.5 \
  --min-samples-split 5 --min-samples-leaf 2 \
  --output-dir log/viper_XO-random --yes

# 3. O-only
python train/train_parallel_viper.py \
  --oracle-path log/oracle_TicTacToe_ppo_aggressive.zip \
  --depth-range 5 25 --depth-step 5 \
  --leaves-range 25 150 --leaves-step 25 \
  --n-iterations 15 --samples-per-iter 10000 \
  --play-as-o-prob 1.0 \
  --min-samples-split 5 --min-samples-leaf 2 \
  --output-dir log/viper_O-only --yes
```

---

## 快速版本（更少配置，用于测试）

如果想快速验证流程，可以使用更小的配置范围：

```bash
# 快速测试: 深度 10-20, 叶子节点 50-100
# 只有 3 × 3 × 3 = 27 个配置

# X-only (快速)
python train/train_parallel_viper.py \
  --oracle-path log/oracle_TicTacToe_ppo_aggressive.zip \
  --depth-range 10 20 --depth-step 5 \
  --leaves-range 50 100 --leaves-step 25 \
  --n-iterations 10 --samples-per-iter 5000 \
  --play-as-o-prob 0.0 \
  --output-dir log/viper_X-only_quick --yes

# XO-random (快速)
python train/train_parallel_viper.py \
  --oracle-path log/oracle_TicTacToe_ppo_aggressive.zip \
  --depth-range 10 20 --depth-step 5 \
  --leaves-range 50 100 --leaves-step 25 \
  --n-iterations 10 --samples-per-iter 5000 \
  --play-as-o-prob 0.5 \
  --output-dir log/viper_XO-random_quick --yes

# O-only (快速)
python train/train_parallel_viper.py \
  --oracle-path log/oracle_TicTacToe_ppo_aggressive.zip \
  --depth-range 10 20 --depth-step 5 \
  --leaves-range 50 100 --leaves-step 25 \
  --n-iterations 10 --samples-per-iter 5000 \
  --play-as-o-prob 1.0 \
  --output-dir log/viper_O-only_quick --yes
```

---

## 时间估算

假设：
- 每个配置训练时间约 10-15 分钟
- 使用 8 个并行进程

**完整训练**（90个配置）:
- 串行时间: 90 × 12分钟 = 1080分钟 ≈ **18小时**
- 并行时间 (8进程): 1080 / 8 ≈ **135分钟 ≈ 2.25小时**

**快速版本**（27个配置）:
- 并行时间: 27 × 8分钟 / 8进程 ≈ **27分钟**

---

## 查看训练结果

### 查看文件结构
```bash
# 查看所有生成的模型
ls -R log/viper_X-only/
ls -R log/viper_XO-random/
ls -R log/viper_O-only/

# 统计每个目录的模型数量
echo "X-only: $(find log/viper_X-only -name "*.joblib" | wc -l) 个模型"
echo "XO-random: $(find log/viper_XO-random -name "*.joblib" | wc -l) 个模型"
echo "O-only: $(find log/viper_O-only -name "*.joblib" | wc -l) 个模型"
```

### 查看配置摘要
```bash
# 查看每个训练批次的配置文件
cat log/viper_X-only/configs_*.json
cat log/viper_XO-random/configs_*.json
cat log/viper_O-only/configs_*.json
```

---

## 训练后评估

训练完成后，可以评估所有模型：

```bash
# 评估所有 X-only 模型 vs MinMax
for model in log/viper_X-only/*.joblib; do
  echo "评估: $model"
  python evaluation/evaluate_viper_tree.py \
    --model-path "$model" \
    --opponent minmax \
    --play-as-o-prob 0.0 \
    --n-episodes 100
done

# 评估所有 XO-random 模型 vs MinMax
for model in log/viper_XO-random/*.joblib; do
  echo "评估: $model"
  python evaluation/evaluate_viper_tree.py \
    --model-path "$model" \
    --opponent minmax \
    --play-as-o-prob 0.5 \
    --n-episodes 100
done

# 评估所有 O-only 模型 vs MinMax
for model in log/viper_O-only/*.joblib; do
  echo "评估: $model"
  python evaluation/evaluate_viper_tree.py \
    --model-path "$model" \
    --opponent minmax \
    --play-as-o-prob 1.0 \
    --n-episodes 100
done
```

---

## 目录结构预览

训练完成后的目录结构（注意目录名自动添加时间戳和参数）：

```
log/
├── viper_X-only_20250117_143022_d5-25_l25-150_iter15/
│   ├── configs_20250117_143022.json
│   ├── viper_mask_ppo_tree_20250117_143022_X-only_iter15_samples10000_depth5_leaves25.joblib
│   ├── viper_mask_ppo_tree_20250117_143022_X-only_iter15_samples10000_depth5_leaves50.joblib
│   ├── ... (共30个 .joblib 文件)
│   └── viper_mask_ppo_tree_20250117_143022_X-only_iter15_samples10000_depth25_leaves150.joblib
│
├── viper_XO-random_20250117_143500_d5-25_l25-150_iter15/
│   ├── configs_20250117_143500.json
│   ├── viper_mask_ppo_tree_20250117_143500_XO-random_iter15_samples10000_depth5_leaves25.joblib
│   ├── ... (共30个 .joblib 文件)
│   └── viper_mask_ppo_tree_20250117_143500_XO-random_iter15_samples10000_depth25_leaves150.joblib
│
└── viper_O-only_20250117_144000_d5-25_l25-150_iter15/
    ├── configs_20250117_144000.json
    ├── viper_mask_ppo_tree_20250117_144000_O-only_iter15_samples10000_depth5_leaves25.joblib
    ├── ... (共30个 .joblib 文件)
    └── viper_mask_ppo_tree_20250117_144000_O-only_iter15_samples10000_depth25_leaves150.joblib
```

**目录命名格式**: `{base_name}_{timestamp}_{depth_range}_{leaves_range}_{iterations}`
- `timestamp`: 训练开始时间 (YYYYmmdd_HHMMSS)
- `depth_range`: 深度范围 (如 d5-25)
- `leaves_range`: 叶子节点范围 (如 l25-150)
- `iterations`: 迭代次数 (如 iter15)

**自动命名的优点**:
- ✓ 时间戳确保每次训练都有独立的目录，不会覆盖
- ✓ 参数信息一目了然，无需打开文件即可知道训练配置
- ✓ 便于比较不同训练配置的结果
- ✓ 支持多次训练实验，易于管理

---

## 注意事项

1. **GPU/CPU使用**:
   - 脚本会自动检测可用的GPU并并行训练
   - 如果没有GPU，会使用CPU（速度较慢）

2. **磁盘空间**:
   - 每个模型约 1-5 MB
   - 90个模型约需 100-500 MB

3. **中断恢复**:
   - 每个 `train_parallel_viper.py` 调用是独立的
   - 如果某个批次失败，可以单独重新运行

4. **监控进度**:
   ```bash
   # 实时查看训练进度
   watch -n 5 'find log/viper_* -name "*.joblib" | wc -l'
   ```

5. **建议顺序**:
   - 先运行快速版本验证流程
   - 再运行完整版本
   - 先运行 X-only，观察结果后再决定是否运行其他配置
