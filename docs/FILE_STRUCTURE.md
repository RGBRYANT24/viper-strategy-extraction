# 项目文件结构说明

本文档说明项目的文件组织结构和各目录用途。

## 目录结构

```
viper-verifiable-rl-impl/
│
├── docs/                          # 📚 文档目录
│   ├── ARCHITECTURE.md           # 项目架构指南 (原 PROJECT_GUIDE.md)
│   ├── MASKABLE_PPO_GUIDE.md     # MaskablePPO 训练指南
│   ├── TRAINING_IMPROVEMENTS.md  # 训练改进策略
│   ├── TESTING_GUIDE.md          # 综合测试指南
│   ├── SYMMETRY_IMPROVEMENT.md   # 对称性改进方法
│   ├── MASKED_DQN_README.md      # 旧版 DQN 文档（已废弃）
│   └── FILE_STRUCTURE.md         # 本文件
│
├── train/                         # 🎯 训练脚本
│   ├── train_delta_selfplay_ppo.py  # ⭐ MaskablePPO 自我对弈训练（主要）
│   ├── oracle.py                 # Oracle 训练基础脚本
│   ├── debug_mask.py             # Action masking 调试脚本
│   └── test_model.py             # 快速模型测试
│
├── evaluation/                    # 📊 评估脚本（新建）
│   ├── evaluate_ppo_strategy.py  # 综合策略评估
│   ├── test_first_second_player.py  # 先后手对比测试
│   └── __init__.py
│
├── gym_env/                       # 🎮 游戏环境
│   ├── __init__.py
│   ├── tictactoe.py              # TicTacToe 基础环境
│   ├── tictactoe_delta_selfplay.py  # 自我对弈环境
│   ├── tictactoe_selfplay.py     # 旧版自我对弈（已废弃）
│   ├── masked_dqn_policy.py      # Masked DQN 策略（已废弃）
│   └── policies/                 # 基准策略
│       ├── __init__.py
│       └── baseline_policies.py  # Random 和 MinMax 策略
│
├── model/                         # 🌲 模型工具
│   ├── rule_extractor.py         # 决策树提取
│   └── tree_wrapper.py           # 决策树策略包装器
│
├── test/                          # ✅ SMT 验证测试（原有）
│   ├── __init__.py
│   ├── evaluate.py               # 评估工具
│   ├── oracle.py                 # Oracle 测试
│   ├── viper.py                  # VIPER 测试
│   └── test_rule_extractor.py    # 规则提取器测试
│
├── log/                           # 💾 保存的模型
│   ├── oracle_TicTacToe_ppo_masked.zip  # 已训练模型
│   ├── oracle_TicTacToe_ppo_balanced.zip
│   └── ...
│
├── archive/                       # 📦 归档脚本（旧版）
│   └── old_scripts/
│
├── main.py                        # 🚀 主入口
├── train_viper_improved.py       # VIPER 训练改进版
└── README.md                      # 项目说明
```

## 主要目录用途

### 📚 `docs/` - 文档

存放所有项目文档，包括架构说明、训练指南、测试指南等。

**关键文档:**
- `TRAINING_IMPROVEMENTS.md`: 训练优化策略，包含参数调优建议
- `TESTING_GUIDE.md`: 完整的测试流程和结果解读
- `ARCHITECTURE.md`: 项目架构和代码组织

### 🎯 `train/` - 训练脚本

包含模型训练相关的脚本。

**主要脚本:**
- `train_delta_selfplay_ppo.py`: **核心训练脚本**，使用 MaskablePPO + 自我对弈
- `test_model.py`: 快速测试模型性能

### 📊 `evaluation/` - 评估脚本

专门用于模型评估和测试的脚本（新建目录）。

**评估工具:**
- `evaluate_ppo_strategy.py`: 综合策略评估（6 个维度）
- `test_first_second_player.py`: 先后手对比测试

### 🎮 `gym_env/` - 游戏环境

包含 TicTacToe 环境实现和对手策略。

**环境文件:**
- `tictactoe.py`: 基础环境（支持 Random 和 MinMax 对手）
- `tictactoe_delta_selfplay.py`: 支持自我对弈的环境
- `policies/baseline_policies.py`: Random 和 MinMax 策略实现

### ✅ `test/` - SMT 验证测试

原有的 SMT 验证和测试代码，用于形式化验证。

**注意:** 不要与新的 `evaluation/` 混淆。

### 💾 `log/` - 模型存储

保存训练好的模型文件。

**命名规范:**
- `oracle_TicTacToe_ppo.zip`: 基础 PPO 模型
- `oracle_TicTacToe_ppo_balanced.zip`: 平衡训练的模型
- `oracle_TicTacToe_ppo_improved.zip`: 改进版模型

## 使用流程

### 1. 训练模型

```bash
# 使用训练脚本
python train/train_delta_selfplay_ppo.py \
    --total-timesteps 300000 \
    --use-minmax \
    --ent-coef 0.05 \
    --random-weight 2.0 \
    --output log/oracle_TicTacToe_ppo_balanced.zip
```

参考文档: [`docs/TRAINING_IMPROVEMENTS.md`](TRAINING_IMPROVEMENTS.md)

### 2. 评估模型

```bash
# 综合策略评估
python evaluation/evaluate_ppo_strategy.py \
    --model log/oracle_TicTacToe_ppo_balanced.zip \
    --num-games 100

# 先后手对比测试
python evaluation/test_first_second_player.py \
    --model log/oracle_TicTacToe_ppo_balanced.zip \
    --num-games 100
```

参考文档: [`docs/TESTING_GUIDE.md`](TESTING_GUIDE.md)

### 3. 提取决策树

```bash
python main.py train-viper --env-name TicTacToe-v0 --n-env 1
```

## 文件移动说明

以下文件已被重新组织：

### 从根目录移到 `docs/`:
- `PROJECT_GUIDE.md` → `docs/ARCHITECTURE.md`
- `MASKABLE_PPO_GUIDE.md` → `docs/MASKABLE_PPO_GUIDE.md`
- `TRAINING_IMPROVEMENTS.md` → `docs/TRAINING_IMPROVEMENTS.md`
- `TESTING_GUIDE.md` → `docs/TESTING_GUIDE.md`
- `SYMMETRY_IMPROVEMENT.md` → `docs/SYMMETRY_IMPROVEMENT.md`
- `MASKED_DQN_README.md` → `docs/MASKED_DQN_README.md`

### 从 `train/` 移到 `evaluation/`:
- `train/evaluate_ppo_strategy.py` → `evaluation/evaluate_ppo_strategy.py`
- `train/test_first_second_player.py` → `evaluation/test_first_second_player.py`

## Git 提交建议

### 提交这些更改

```bash
# 添加新文件
git add docs/
git add evaluation/

# 移除旧路径（Git 会自动检测移动）
git add -A

# 提交
git commit -m "Reorganize project structure: move docs to docs/ and tests to evaluation/"
```

### 提交信息模板

```
Reorganize project structure

Changes:
- Created docs/ directory for all documentation
- Created evaluation/ directory for evaluation scripts
- Moved markdown files from root to docs/
- Renamed PROJECT_GUIDE.md to ARCHITECTURE.md for clarity
- Moved evaluate_ppo_strategy.py and test_first_second_player.py to evaluation/
- test/ directory remains for SMT verification (original purpose)

This improves project organization and separates concerns:
- docs/ for documentation
- train/ for training scripts
- evaluation/ for model evaluation
- test/ for SMT verification (original framework)
```

## 注意事项

1. **`test/` 目录**: 保留原有用途（SMT 验证），不用于模型评估
2. **`evaluation/` 目录**: 新建目录，专门用于模型评估
3. **文档路径**: 所有文档引用需要更新为 `docs/` 下的路径
4. **导入路径**: Python 导入路径无需修改（评估脚本使用绝对导入）

## 快速参考

| 任务 | 脚本位置 | 文档位置 |
|------|---------|---------|
| 训练模型 | `train/train_delta_selfplay_ppo.py` | `docs/TRAINING_IMPROVEMENTS.md` |
| 评估策略 | `evaluation/evaluate_ppo_strategy.py` | `docs/TESTING_GUIDE.md` |
| 测试先后手 | `evaluation/test_first_second_player.py` | `docs/TESTING_GUIDE.md` |
| 理解架构 | - | `docs/ARCHITECTURE.md` |
| 对称性改进 | - | `docs/SYMMETRY_IMPROVEMENT.md` |
