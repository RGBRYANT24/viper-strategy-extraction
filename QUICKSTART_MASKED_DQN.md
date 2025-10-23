# Masked DQN 快速开始指南

## 一行命令测试

```bash
# 1. 测试 MaskedDQNPolicy（30秒）
python3 gym_env/masked_dqn_policy.py

# 2. 快速训练测试（2分钟）
python3 train/train_delta_selfplay.py --total-timesteps 20000 --n-env 4 --output log/test.zip

# 3. 评估模型质量（1分钟）
python3 evaluate_nn_quality.py --model log/test.zip
```

## 完整训练流程

### 在远程服务器上运行

```bash
# SSH连接到服务器
ssh your-server

# 进入项目目录
cd /path/to/viper-verifiable-rl-impl

# 激活虚拟环境（如果有）
# source venv/bin/activate

# 运行完整训练（20分钟）
python3 train/train_delta_selfplay.py \
    --total-timesteps 200000 \
    --n-env 8 \
    --use-minmax \
    --max-pool-size 20 \
    --output log/oracle_TicTacToe_masked.zip \
    --verbose 1

# 评估模型
python3 evaluate_nn_quality.py \
    --model log/oracle_TicTacToe_masked.zip

# 提取决策树（VIPER）
python3 main.py \
    --train-viper \
    --env-name TicTacToe-v0 \
    --oracle log/oracle_TicTacToe_masked.zip \
    --max-depth 10
```

## 后台运行（推荐）

使用 `nohup` 或 `screen` 在后台运行：

```bash
# 方法1：使用 nohup
nohup python3 train/train_delta_selfplay.py \
    --total-timesteps 200000 \
    --n-env 8 \
    --use-minmax \
    --output log/oracle_TicTacToe_masked.zip \
    > train.log 2>&1 &

# 查看日志
tail -f train.log

# 方法2：使用 screen
screen -S train_dqn
python3 train/train_delta_selfplay.py --total-timesteps 200000 --use-minmax
# 按 Ctrl+A 再按 D 离开screen
# 重新连接：screen -r train_dqn
```

## 预期结果检查清单

### ✓ 训练成功的标志
- [ ] 训练过程中 `非法移动: 0`
- [ ] 最终测试 `平局率 >= 80%`
- [ ] 输出日志中看到 `✓ 使用 MaskedDQNPolicy`
- [ ] 模型文件生成：`log/oracle_TicTacToe_masked.zip`

### ✓ 评估成功的标志
- [ ] Action Masking 评估：`✓ 通过`
- [ ] 关键局面识别：`>= 90%`
- [ ] vs MinMax 平局率：`>= 95%`
- [ ] 非法移动：`0`

### ✓ VIPER 成功的标志
- [ ] 决策树文件生成
- [ ] 策略准确率 >= 95%
- [ ] 树的深度合理（通常 5-15 层）

## 常见问题快速解决

### Q1: `ModuleNotFoundError: No module named 'gym_env.masked_dqn_policy'`

```bash
# 确保在项目根目录
cd /path/to/viper-verifiable-rl-impl

# 检查文件是否存在
ls -la gym_env/masked_dqn_policy.py

# 如果不存在，检查git状态
git status
```

### Q2: 训练中出现非法移动

```bash
# 1. 重新测试policy
python3 gym_env/masked_dqn_policy.py

# 2. 检查模型是否使用了MaskedDQNPolicy
python3 -c "
from stable_baselines3 import DQN
from gym_env.masked_dqn_policy import MaskedDQNPolicy
model = DQN.load('log/oracle_TicTacToe_masked.zip')
print('是否Masked:', isinstance(model.policy, MaskedDQNPolicy))
"
```

### Q3: 平局率太低（< 80%）

```bash
# 增加训练步数和MinMax对手
python3 train/train_delta_selfplay.py \
    --total-timesteps 300000 \
    --use-minmax \
    --max-pool-size 30 \
    --output log/oracle_TicTacToe_longer.zip
```

## 参数调优建议

### 快速实验（开发阶段）
```bash
python3 train/train_delta_selfplay.py \
    --total-timesteps 20000 \
    --n-env 4 \
    --update-interval 5000 \
    --max-pool-size 5
```

### 标准训练（生产环境）
```bash
python3 train/train_delta_selfplay.py \
    --total-timesteps 200000 \
    --n-env 8 \
    --use-minmax \
    --update-interval 10000 \
    --max-pool-size 20
```

### 高质量训练（追求最优）
```bash
python3 train/train_delta_selfplay.py \
    --total-timesteps 500000 \
    --n-env 16 \
    --use-minmax \
    --update-interval 10000 \
    --max-pool-size 50 \
    --net-arch 256,256
```

## 监控训练进度

### 实时监控
```bash
# 方法1：使用watch
watch -n 10 'tail -n 20 train.log'

# 方法2：使用grep筛选关键信息
tail -f train.log | grep -E "(训练轮次|平局率|非法移动)"

# 方法3：检查模型文件大小（确认在保存）
watch -n 30 'ls -lh log/oracle_TicTacToe_masked.zip'
```

### 中断后继续训练
```bash
# DQN不支持从checkpoint继续，但可以：
# 1. 加载已有模型继续训练
python3 -c "
from stable_baselines3 import DQN
model = DQN.load('log/oracle_TicTacToe_masked.zip')
# 继续训练
model.learn(total_timesteps=100000, reset_num_timesteps=False)
model.save('log/oracle_TicTacToe_continued.zip')
"
```

## 下一步

训练完成后，参考 [MASKED_DQN_README.md](MASKED_DQN_README.md) 进行：
- 详细的模型评估
- VIPER 决策树提取
- 形式化验证

## 获取帮助

```bash
# 查看训练脚本帮助
python3 train/train_delta_selfplay.py --help

# 查看评估脚本帮助
python3 evaluate_nn_quality.py --help
```
