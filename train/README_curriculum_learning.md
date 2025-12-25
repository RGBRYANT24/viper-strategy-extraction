# 课程学习训练：提升关键决策能力

## 核心思路

使用**课程学习（Curriculum Learning）**，在保持 Delta-Uniform Self-play 的基础上，混入关键决策场景进行训练。

### 原理

- **80-50% Self-play** + **20-50% 关键场景**
- 保持原有的奖励机制（+1获胜/-1失败/0平局/-10非法）
- 保持 Delta-Uniform 对手采样策略
- 逐步调整关键场景比例（课程）

### 为什么有效？

你的模型当前问题：
- 总准确率 78%（不够好）
- 防守能力弱（73%）
- 特定位置容易出错

**原因**：Self-play 训练时，很少遇到这些关键场景（概率太低）

**解决方案**：
1. **主动注入**关键场景，增加学习机会
2. **保持 Self-play**，不影响整体对战能力
3. **课程式调整**，先多练后巩固

## 使用方法

### 基本用法

```bash
# 推荐配置：从你的模型开始
python train/train_curriculum_learning.py \
    --model log/oracle_TicTacToe_ppo_aggressive.zip \
    --total-timesteps 150000 \
    --initial-critical-prob 0.2 \
    --max-critical-prob 0.5 \
    --ent-coef 0.05
```

### 参数说明

#### 必需参数

- `--model <路径>`: 要加载的模型（支持你已训练好的任何模型）

#### 课程学习参数（关键！）

| 参数 | 默认值 | 推荐值 | 说明 |
|-----|--------|--------|------|
| `--initial-critical-prob` | 0.2 | 0.2-0.3 | 初始关键场景比例 |
| `--max-critical-prob` | 0.5 | 0.4-0.6 | 最大关键场景比例 |

#### 训练参数

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `--total-timesteps` | 150000 | 总训练步数 |
| `--n-env` | 8 | 并行环境数 |
| `--update-interval` | 10000 | 更新策略池间隔 |
| `--max-pool-size` | 20 | 策略池容量 |

#### PPO 参数

| 参数 | 默认值 | 推荐值 | 说明 |
|-----|--------|--------|------|
| `--ent-coef` | 0.05 | **0.05** | 熵系数（保持原值） |
| `--play-as-o-prob` | 0.5 | 0.5 | 作为O方概率 |
| `--random-weight` | 2.0 | 2.0 | Random对手权重 |

### 课程计划

训练过程中会自动调整关键场景比例：

```
步数       关键场景比例    说明
0         20%            开始，温和引入
37500     35%            逐步增加
75000     50%            最大强度
112500    35%            开始回归
135000    20%            回到初始，巩固
```

**为什么这样设计？**
- **前期（0-25%）**：温和引入，避免破坏已有知识
- **中期（25-75%）**：最大化关键场景，集中学习
- **后期（75-100%）**：逐步回归，确保泛化能力

## 使用示例

### 推荐流程

#### 1. 从你当前的模型开始

```bash
python train/train_curriculum_learning.py \
    --model log/oracle_TicTacToe_ppo.zip \
    --total-timesteps 150000 \
    --initial-critical-prob 0.2 \
    --max-critical-prob 0.5 \
    --ent-coef 0.05
```

**预期**：
- 训练时间：约15-30分钟（取决于硬件）
- 准确率提升：78% → 85-90%

#### 2. 测试结果

```bash
python evaluation/exhaustive_win_lose_test.py \
    --model log/oracle_TicTacToe_ppo_curriculum_cr50_steps150k_<时间戳>.zip
```

#### 3. 如果还不够好，增加强度

```bash
# 更高的关键场景比例
python train/train_curriculum_learning.py \
    --model log/oracle_TicTacToe_ppo_curriculum_cr50_steps150k_<时间戳>.zip \
    --total-timesteps 100000 \
    --initial-critical-prob 0.3 \
    --max-critical-prob 0.6 \
    --ent-coef 0.05
```

### 高级配置

#### 激进训练（如果基础训练效果不明显）

```bash
python train/train_curriculum_learning.py \
    --model log/oracle_TicTacToe_ppo.zip \
    --total-timesteps 200000 \
    --initial-critical-prob 0.3 \
    --max-critical-prob 0.7 \
    --ent-coef 0.05
```

#### 保守训练（如果担心过拟合）

```bash
python train/train_curriculum_learning.py \
    --model log/oracle_TicTacToe_ppo.zip \
    --total-timesteps 150000 \
    --initial-critical-prob 0.15 \
    --max-critical-prob 0.4 \
    --ent-coef 0.05
```

## 输出文件名

自动生成，格式：
```
<原模型名>_curriculum_cr<最大比例>_steps<步数>k_<时间戳>.zip
```

例如：
```
oracle_TicTacToe_ppo_curriculum_cr50_steps150k_20250104_153022.zip
```

- `cr50`：最大关键场景比例50%
- `steps150k`：训练150k步

## 与其他方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| **提高熵系数** | 简单 | 可能破坏已有知识 | 模型表现平庸 |
| **纯 Self-play** | 泛化好 | 关键场景太少 | 初始训练 |
| **课程学习（本方法）** | **针对性强，保持泛化** | **略复杂** | **提升特定能力** |

## 为什么比提高熵系数好？

你之前尝试 ent-coef=0.15，结果准确率反而下降了（78.27% → 77.68%）。

**原因分析**：
- ❌ 熵系数过高 → 策略过于随机 → 破坏已学知识
- ❌ Self-play中关键场景概率仍然很低

**课程学习的优势**：
- ✅ **不提高熵系数**（保持0.05），保护已有知识
- ✅ **主动注入关键场景**，学习机会多100倍+
- ✅ **保持Self-play**，不影响整体对战能力

## 预期效果

根据课程学习原理和你的模型现状：

### 保守估计（initial=0.2, max=0.5, 150k步）

| 指标 | 训练前 | 预期训练后 |
|-----|--------|-----------|
| 总准确率 | 78% | **85-88%** |
| 获胜准确率 | 81% | **88-92%** |
| 防守准确率 | 75% | **82-85%** |

### 乐观估计（initial=0.3, max=0.6, 200k步）

| 指标 | 训练前 | 预期训练后 |
|-----|--------|-----------|
| 总准确率 | 78% | **90-93%** |
| 获胜准确率 | 81% | **92-95%** |
| 防守准确率 | 75% | **88-92%** |

## 训练监控

训练过程中会显示：
1. 当前训练步数
2. 课程调整提示（每次比例变化）
3. 策略池更新信息

示例输出：
```
======================================================================
[课程学习] 第 37500 步
调整关键场景比例: 35%
======================================================================
```

## 注意事项

### 1. 熵系数建议保持原值

不要提高熵系数！课程学习已经提供了足够的探索。

```bash
# ✓ 推荐
--ent-coef 0.05

# ✗ 不推荐（会破坏已有知识）
--ent-coef 0.1 或更高
```

### 2. 训练步数建议

- **最少**：100k步（可能效果有限）
- **推荐**：150-200k步
- **充分**：200k+步（如果时间允许）

### 3. 关键场景比例

- **太低（max<0.4）**：效果可能不明显
- **推荐（max=0.4-0.6）**：平衡效果和泛化
- **太高（max>0.7）**：可能过拟合关键场景

### 4. 对战能力

课程学习后，建议也测试整体对战能力：

```bash
# 测试关键决策
python evaluation/exhaustive_win_lose_test.py --model <模型>

# 测试整体策略
python evaluation/evaluate_ppo_strategy.py --model <模型> --num-games 100
```

## 故障排查

### 如果训练后准确率没提升

1. **增加关键场景比例**：
   ```bash
   --max-critical-prob 0.6
   ```

2. **增加训练步数**：
   ```bash
   --total-timesteps 200000
   ```

3. **检查熵系数**：
   - 确保使用 `--ent-coef 0.05`（不要更高）

### 如果整体对战能力下降

说明关键场景比例太高，降低比例：
```bash
--max-critical-prob 0.4
```

或者在课程训练后，再用少量步数做纯Self-play巩固：
```bash
python train/train_delta_selfplay_ppo.py \
    --load-model <课程训练后的模型> \
    --total-timesteps 50000 \
    --ent-coef 0.05 \
    --auto-name
```

## 完整训练流程示例

```bash
# 1. 课程学习训练（150k步）
python train/train_curriculum_learning.py \
    --model log/oracle_TicTacToe_ppo.zip \
    --total-timesteps 150000 \
    --initial-critical-prob 0.2 \
    --max-critical-prob 0.5 \
    --ent-coef 0.05

# 2. 测试关键决策能力
python evaluation/exhaustive_win_lose_test.py \
    --model log/oracle_TicTacToe_ppo_curriculum_cr50_steps150k_20250104_153022.zip

# 3. 如果满意，测试整体能力
python evaluation/evaluate_ppo_strategy.py \
    --model log/oracle_TicTacToe_ppo_curriculum_cr50_steps150k_20250104_153022.zip \
    --num-games 100

# 4. 如果还需要提升，继续课程训练
python train/train_curriculum_learning.py \
    --model log/oracle_TicTacToe_ppo_curriculum_cr50_steps150k_20250104_153022.zip \
    --total-timesteps 100000 \
    --initial-critical-prob 0.3 \
    --max-critical-prob 0.6 \
    --ent-coef 0.05
```

## 总结

课程学习是针对你当前问题的**最佳方案**：

✅ **针对性强**：重点训练关键决策
✅ **保护已有知识**：不提高熵系数
✅ **保持泛化**：仍包含Self-play
✅ **可控性好**：可调整关键场景比例

相比之下：
- ❌ 提高熵系数：会破坏已有知识（你已经验证过）
- ❌ 纯Self-play：遇到关键场景太少

**建议立即开始尝试课程学习！**
