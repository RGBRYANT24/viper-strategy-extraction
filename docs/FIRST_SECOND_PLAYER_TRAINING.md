# VIPER 训练 - 先后手配置指南

## 快速开始

### 参数说明

**`--play-as-o-prob`**: 作为后手(O)的概率
- `0.0`: 只学习先手(X) → 文件名包含 `X-only`
- `0.5`: 随机先后手(默认) → 文件名包含 `XO-random`
- `1.0`: 只学习后手(O) → 文件名包含 `O-only`

## 训练示例

### 1. 只训练先手（推荐优先尝试）

```bash
python train/viper_mask_ppo.py \
  --mode train \
  --oracle_path log/oracle_TicTacToe_ppo_aggressive.zip \
  --output_path viper_tree \
  --n_iterations 15 \
  --samples_per_iter 10000 \
  --max_depth 15 \
  --max_leaves 100 \
  --play-as-o-prob 0.0
```

**生成文件**: `viper_tree_20250117_123456_X-only_iter15_samples10000_depth15_leaves100.joblib`

### 2. 先后手随机

```bash
python train/viper_mask_ppo.py \
  --mode train \
  --oracle_path log/oracle_TicTacToe_ppo_aggressive.zip \
  --output_path viper_tree \
  --n_iterations 15 \
  --samples_per_iter 10000 \
  --max_depth 20 \
  --max_leaves 150 \
  --play-as-o-prob 0.5
```

**生成文件**: `viper_tree_20250117_123456_XO-random_iter15_samples10000_depth20_leaves150.joblib`

### 3. 只训练后手

```bash
python train/viper_mask_ppo.py \
  --mode train \
  --oracle_path log/oracle_TicTacToe_ppo_aggressive.zip \
  --output_path viper_tree \
  --n_iterations 15 \
  --samples_per_iter 10000 \
  --max_depth 15 \
  --max_leaves 100 \
  --play-as-o-prob 1.0
```

**生成文件**: `viper_tree_20250117_123456_O-only_iter15_samples10000_depth15_leaves100.joblib`

## 并行训练示例

### 只训练先手 + 网格搜索

```bash
python train/train_parallel_viper.py \
  --oracle-path log/oracle_TicTacToe_ppo_aggressive.zip \
  --depth-range 10 20 \
  --depth-step 5 \
  --leaves-range 50 150 \
  --leaves-step 25 \
  --n-iterations 15 \
  --samples-per-iter 10000 \
  --play-as-o-prob 0.0 \
  --yes
```

## 文件命名规则

生成的文件名格式：`{base_name}_{timestamp}_{player_str}_{params}.joblib`

其中 `player_str` 为：
- `X-only`: 只学习先手
- `XO-random`: 随机先后手
- `O-only`: 只学习后手
- `O-prob30`: 30% 概率作为后手（自定义概率）

## 推荐配置

| 场景 | play-as-o-prob | max-depth | max-leaves | samples-per-iter |
|------|----------------|-----------|------------|------------------|
| 快速验证 | 0.0 | 10 | 50 | 5000 |
| 先手策略 | 0.0 | 15 | 100 | 10000 |
| 全面学习 | 0.5 | 20 | 150 | 10000 |
| 后手策略 | 1.0 | 15 | 100 | 10000 |

## 决策流程

1. **测试 Oracle 性能**:
   ```bash
   python test/test_oracle_performance.py \
     --oracle-path log/oracle_TicTacToe_ppo_aggressive.zip \
     --opponent minmax --play-as-o-prob 0.5 --n-episodes 100
   ```

2. **根据结果选择训练策略**:
   - 如果 Oracle 平局率 > 75%: 使用 `--play-as-o-prob 0.5`
   - 如果 Oracle 平局率 < 75%: 使用 `--play-as-o-prob 0.0`

3. **训练并评估**:
   ```bash
   # 训练
   python train/viper_mask_ppo.py --mode train [参数...]

   # 评估
   python evaluation/evaluate_viper_tree.py \
     --model-path log/viper_*.joblib \
     --opponent minmax --n-episodes 100
   ```
