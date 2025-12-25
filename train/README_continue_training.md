# 继续训练已有模型

修改后的训练脚本支持加载已有模型继续训练，并提供自动版本管理。

## 使用方法

### 1. 继续训练已有模型（提高探索性）

根据你的测试结果（总正确率 78.27%），建议提高熵系数以增加探索：

```bash
# 基本用法：加载模型并提高熵系数到 0.1
python train/train_delta_selfplay_ppo.py \
    --load-model log/oracle_TicTacToe_ppo_aggressive.zip \
    --ent-coef 0.1 \
    --total-timesteps 100000 \
    --auto-name

# 更激进的探索（熵系数 0.15）
python train/train_delta_selfplay_ppo.py \
    --load-model log/oracle_TicTacToe_ppo_aggressive.zip \
    --ent-coef 0.15 \
    --total-timesteps 100000 \
    --auto-name

# 最激进的探索（熵系数 0.2）- 如果前面效果不好再用
python train/train_delta_selfplay_ppo.py \
    --load-model log/oracle_TicTacToe_ppo_aggressive.zip \
    --ent-coef 0.2 \
    --total-timesteps 100000 \
    --auto-name
```

### 2. 自动命名说明

使用 `--auto-name` 参数时，输出文件名会自动生成，格式为：

**继续训练时**：
```
<原模型名>_cont_ent<熵系数>_steps<步数>k_<时间戳>.zip
```

例如：
```
oracle_TicTacToe_ppo_cont_ent0.1_steps100k_20250104_143022.zip
```

**从头训练时**：
```
oracle_TicTacToe_ppo_ent<熵系数>_steps<步数>k_<时间戳>.zip
```

### 3. 完整参数示例

```bash
python train/train_delta_selfplay_ppo.py \
    --load-model log/oracle_TicTacToe_ppo.zip \
    --ent-coef 0.1 \
    --total-timesteps 100000 \
    --n-env 8 \
    --update-interval 10000 \
    --max-pool-size 20 \
    --random-weight 2.0 \
    --auto-name
```

## 参数说明

### 新增参数

- `--load-model <路径>`: 加载已有模型继续训练
- `--auto-name`: 自动生成带版本信息的输出文件名

### 关键调优参数

| 参数 | 默认值 | 建议值 | 说明 |
|-----|--------|--------|------|
| `--ent-coef` | 0.05 | **0.1-0.2** | 熵系数，控制探索性。数值越大，探索越多 |
| `--total-timesteps` | 200000 | 50000-150000 | 继续训练的步数 |
| `--random-weight` | 2.0 | 2.0-3.0 | Random对手的采样权重，增加多样性 |

### 其他参数

- `--n-env`: 并行环境数（默认: 8）
- `--update-interval`: 更新策略池的间隔（默认: 10000）
- `--max-pool-size`: 策略池容量（默认: 20）
- `--output`: 手动指定输出路径（不使用 `--auto-name` 时）

## 训练策略建议

### 针对你的测试结果

测试显示：
- 获胜案例准确率: 83.53%（还不错）
- 防守案例准确率: 73.02%（需要改进）
- 总体: 78.27%

**建议的训练方案**：

#### 方案1：中等探索（推荐优先尝试）
```bash
python train/train_delta_selfplay_ppo.py \
    --load-model log/oracle_TicTacToe_ppo.zip \
    --ent-coef 0.1 \
    --total-timesteps 100000 \
    --auto-name
```

训练完成后测试：
```bash
python evaluation/exhaustive_win_lose_test.py \
    --model log/oracle_TicTacToe_ppo_cont_ent0.1_steps100k_<时间戳>.zip
```

期望结果：总正确率提升到 85-90%

#### 方案2：高探索（如果方案1效果不够）
```bash
python train/train_delta_selfplay_ppo.py \
    --load-model log/oracle_TicTacToe_ppo_cont_ent0.1_steps100k_<时间戳>.zip \
    --ent-coef 0.15 \
    --total-timesteps 100000 \
    --auto-name
```

期望结果：总正确率提升到 90-95%

#### 方案3：非常高的探索（最后的手段）
```bash
python train/train_delta_selfplay_ppo.py \
    --load-model <上一次模型> \
    --ent-coef 0.2 \
    --total-timesteps 100000 \
    --auto-name
```

⚠️ 注意：过高的熵系数可能导致训练不稳定

## 为什么提高熵系数有用？

你的模型当前问题：
- 防守能力较弱（73.02%）
- 在某些位置（Col2, Col0, Row2, Row0）容易出错

**原因**：可能是训练时探索不足，模型没有充分学习到这些关键场景

**解决方案**：
- 提高熵系数 → 增加随机性 → 更多探索不同的棋局
- Self-play → 随着策略池更新，会遇到更多关键决策场景
- 更多训练步数 → 有更多机会学习防守

## 训练监控

训练过程中会显示：
- 当前训练步数
- 策略池大小
- 定期对战 MinMax 的结果（在测试部分）

## 完整流程示例

```bash
# 1. 加载当前模型，提高探索，训练 100k 步
python train/train_delta_selfplay_ppo.py \
    --load-model log/oracle_TicTacToe_ppo.zip \
    --ent-coef 0.1 \
    --total-timesteps 100000 \
    --auto-name

# 2. 测试新模型
python evaluation/exhaustive_win_lose_test.py \
    --model log/oracle_TicTacToe_ppo_cont_ent0.1_steps100k_20250104_143022.zip

# 3. 如果结果好，保存这个版本；如果不够好，继续训练
python train/train_delta_selfplay_ppo.py \
    --load-model log/oracle_TicTacToe_ppo_cont_ent0.1_steps100k_20250104_143022.zip \
    --ent-coef 0.15 \
    --total-timesteps 100000 \
    --auto-name
```

## 对比测试

为了看到改进效果，建议对比测试：

```bash
# 测试原始模型
python evaluation/exhaustive_win_lose_test.py \
    --model log/oracle_TicTacToe_ppo.zip

# 测试新模型
python evaluation/exhaustive_win_lose_test.py \
    --model log/oracle_TicTacToe_ppo_cont_ent0.1_steps100k_<时间戳>.zip
```

对比两个模型的：
- 总正确率
- 获胜案例准确率
- 防守案例准确率
- 错误分布（哪些位置改进了）

## 注意事项

1. **备份原模型**：虽然使用 `--auto-name` 会自动创建新文件，但建议备份原模型
2. **逐步调整**：先尝试中等探索（0.1），再逐步提高
3. **监控训练**：注意 vs MinMax 的对战结果，确保模型没有退化
4. **版本管理**：自动生成的文件名已包含参数和时间戳，方便追踪

## 预期改进

经过 100k 步训练（ent-coef=0.1）：
- 总正确率: 78% → 85%+
- 防守准确率: 73% → 80%+
- 错误模式: 某些位置的错误显著减少

如果需要更好的结果，可以继续训练或提高熵系数。
