# TicTacToe 训练指南

## 问题诊断

当前训练方案存在的问题：
- **神经网络对手**：随机玩家（Random Player）
- **结果**：决策树 vs MinMax 胜率 0%（0胜100负）
- **原因**：神经网络只学会了"比随机好一点"的策略，没有学到最优策略

## 改进方案：使用MinMax对手训练

### 方案对比

| 训练方案 | 对手类型 | 优点 | 缺点 | 预期效果 |
|---------|---------|------|------|---------|
| **原方案** | Random | 训练快速，容易收敛 | 策略质量低 | 决策树 vs MinMax: 0% |
| **改进方案** | MinMax | 学到最优策略 | 训练困难，需要更多步数 | 决策树 vs MinMax: >80% 平局 |

---

## 快速开始

### 1. 使用MinMax对手训练（推荐）

```bash
# 给脚本添加执行权限
chmod +x train_with_minmax.sh

# 运行完整训练流程
./train_with_minmax.sh
```

### 2. 手动训练（更灵活）

#### 步骤 1: 训练神经网络对战MinMax

```bash
python main.py train-oracle \
    --env-name TicTacToe-v0 \
    --total-timesteps 100000 \
    --n-env 8 \
    --seed 42 \
    --tictactoe-opponent minmax \
    --oracle-path log/oracle_TicTacToe-v0_minmax.zip
```

**重要参数说明：**
- `--tictactoe-opponent minmax`: 使用MinMax作为对手
- `--total-timesteps 100000`: 增加训练步数（MinMax更难对付）
- `--tictactoe-minmax-depth 9`: MinMax完全搜索（可选，默认为9）

#### 步骤 2: 测试神经网络

```bash
python main.py test-oracle \
    --env-name TicTacToe-v0 \
    --n-env 1 \
    --tictactoe-opponent minmax \
    --oracle-path log/oracle_TicTacToe-v0_minmax.zip
```

**预期结果：**
- 平局率应该 >80%（说明学到了接近最优的策略）
- 胜率低（因为MinMax是最优的）
- 如果败率很高，需要增加训练步数

#### 步骤 3: 提取决策树

```bash
python main.py train-viper \
    --env-name TicTacToe-v0 \
    --n-iter 80 \
    --n-env 8 \
    --seed 42 \
    --tictactoe-opponent minmax \
    --oracle-path log/oracle_TicTacToe-v0_minmax.zip
```

#### 步骤 4: 验证决策树质量

```bash
python battle_nn_vs_tree.py --mode all --n-games 100 \
    --oracle-path log/oracle_TicTacToe-v0_minmax.zip \
    --viper-path log/viper_TicTacToe-v0_all-leaves_80.joblib
```

**预期结果：**
- 神经网络 vs 决策树：高平局率（>80%）说明决策树学到了神经网络的策略
- 决策树 vs MinMax：平局率应该显著提高（从0%提升到>60%）
- 神经网络 vs MinMax：平局率 >80%

---

## 训练选项对比

### 选项 1: 完全MinMax（深度=9）

```bash
--tictactoe-opponent minmax --tictactoe-minmax-depth 9
```

- **优点**：学到绝对最优策略
- **缺点**：训练最慢
- **推荐场景**：最终模型训练

### 选项 2: 部分MinMax（深度=5-7）

```bash
--tictactoe-opponent minmax --tictactoe-minmax-depth 5
```

- **优点**：训练速度快
- **缺点**：策略不是完全最优
- **推荐场景**：快速实验、调参

### 选项 3: 随机对手（原方案）

```bash
--tictactoe-opponent random
```

- **优点**：训练极快
- **缺点**：策略质量低
- **推荐场景**：仅用于测试框架

---

## 训练超参数建议

### 对战Random对手

```bash
--total-timesteps 10000
--n-env 8
--n-iter 80
```

### 对战MinMax对手（推荐）

```bash
--total-timesteps 100000   # 增加10倍
--n-env 8
--n-iter 100               # 增加VIPER迭代
```

### 对战弱MinMax对手（快速实验）

```bash
--total-timesteps 50000
--tictactoe-minmax-depth 5  # 减少搜索深度
--n-env 8
--n-iter 80
```

---

## 训练进度监控

### 检查训练是否成功

1. **神经网络测试**
   ```bash
   python main.py test-oracle \
       --env-name TicTacToe-v0 \
       --tictactoe-opponent minmax \
       --oracle-path log/oracle_TicTacToe-v0_minmax.zip
   ```
   - 平局率 >80% = 优秀
   - 平局率 60-80% = 良好
   - 平局率 <60% = 需要更多训练

2. **决策树验证**
   ```bash
   python battle_nn_vs_tree.py --mode all --n-games 100 \
       --oracle-path log/oracle_TicTacToe-v0_minmax.zip \
       --viper-path log/viper_TicTacToe-v0_all-leaves_80.joblib
   ```

### 常见问题

**Q: 训练很慢怎么办？**
- 减少 `--tictactoe-minmax-depth` (例如从9改到5)
- 减少 `--total-timesteps`
- 增加 `--n-env` 并行环境数

**Q: 神经网络对MinMax胜率很低？**
- 正常！MinMax是最优策略，神经网络很难战胜它
- 关注平局率，平局率高说明学得好

**Q: 决策树 vs MinMax 还是输？**
- 检查神经网络本身的表现
- 增加VIPER迭代次数 `--n-iter`
- 考虑增加决策树深度 `--max-depth`

**Q: 神经网络 vs 决策树平局率低？**
- 说明决策树没有很好地学到神经网络的策略
- 增加VIPER训练数据量和迭代次数
- 检查决策树是否过拟合或欠拟合

---

## 完整示例命令

### 场景 1: 快速验证（5分钟内）

```bash
# 训练神经网络（使用弱MinMax）
python main.py train-oracle \
    --env-name TicTacToe-v0 \
    --total-timesteps 20000 \
    --tictactoe-opponent minmax \
    --tictactoe-minmax-depth 3 \
    --oracle-path log/oracle_quick.zip

# 提取决策树
python main.py train-viper \
    --env-name TicTacToe-v0 \
    --n-iter 40 \
    --tictactoe-opponent minmax \
    --tictactoe-minmax-depth 3 \
    --oracle-path log/oracle_quick.zip

# 测试
python battle_nn_vs_tree.py --mode all --n-games 50 \
    --oracle-path log/oracle_quick.zip
```

### 场景 2: 高质量训练（30-60分钟）

```bash
# 训练神经网络（完全MinMax）
python main.py train-oracle \
    --env-name TicTacToe-v0 \
    --total-timesteps 200000 \
    --tictactoe-opponent minmax \
    --tictactoe-minmax-depth 9 \
    --oracle-path log/oracle_best.zip

# 提取决策树
python main.py train-viper \
    --env-name TicTacToe-v0 \
    --n-iter 120 \
    --tictactoe-opponent minmax \
    --oracle-path log/oracle_best.zip

# 全面测试
python battle_nn_vs_tree.py --mode all --n-games 500 \
    --oracle-path log/oracle_best.zip
```

---

## 总结

使用MinMax对手训练的优势：

✅ **学到最优策略**：神经网络能达到接近最优的表现
✅ **决策树质量高**：模仿更好的策略，自然效果更好
✅ **可验证性强**：与MinMax对比，可以量化评估策略质量
✅ **泛化能力好**：学到的是真正的游戏策略，而不是"如何战胜随机玩家"

推荐训练流程：
1. 使用 `train_with_minmax.sh` 一键训练
2. 使用 `battle_nn_vs_tree.py` 验证三方对战
3. 观察平局率作为策略质量的主要指标
