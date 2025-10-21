# 修复总结

## 修复的问题

### 问题 1: 向量化环境索引错误
**错误信息**：
```
IndexError: invalid index to scalar variable.
```

**原因**：
- 决策树预测时返回标量，但向量化环境需要数组
- 在第2次迭代使用决策树策略时，`action` 变成标量导致 `env.step(action)` 失败

**修复**：
- 在循环开始前检测是否为向量化环境
- 决策树预测时，如果是向量化环境使用 `policy.predict(obs)` 而不是 `policy.predict(obs[0].reshape(1, -1))[0]`
- 随机探索时也生成正确维度的动作数组

### 问题 2: 参数配置不合理
**原始配置问题**：
- `max-leaves=50`：太小，第一次迭代就达到上限
- 评估得分 -2.67：策略质量很差
- 迭代80次：可能过多

**新的推荐配置**：
```bash
--n-iter 60              # 60次迭代（原80）
--max-leaves 100         # 100个叶子（原50）
--max-depth 15           # 深度15（原10）
--ccp-alpha 0.0005       # 更小的正则化（原0.001）
--min-samples-split 5    # 更容易分裂（原10）
--min-samples-leaf 2     # 更小的叶子（原5）
```

## 上传修复文件

**在本地执行**：
```bash
# 方法1: 只上传修复的文件（最快）
scp /Users/adrin/Projects/NOGGINGS/VIPER/viper-verifiable-rl-impl/train_viper_improved.py user@server:~/viper/

# 方法2: 上传所有更新的文件
cd /Users/adrin/Projects/NOGGINGS/VIPER/viper-verifiable-rl-impl
scp train_viper_improved.py QUICK_COMMANDS.md run_improved_viper.sh FIX_SUMMARY.md user@server:~/viper/
```

## 在服务器上重新训练

```bash
# 登录服务器
ssh user@server
cd ~/viper
conda activate your_env

# 使用新配置训练
python train_viper_improved.py \
  --env-name TicTacToe-v0 \
  --oracle-path log/oracle_TicTacToe_selfplay.zip \
  --n-iter 60 \
  --total-timesteps 100000 \
  --max-leaves 100 \
  --max-depth 15 \
  --use-augmentation \
  --exploration-strategy decay \
  --ccp-alpha 0.0005 \
  --min-samples-split 5 \
  --min-samples-leaf 2 \
  --verbose 1
```

## 预期改进

**第一次迭代**：
- ✅ 不会立即达到叶子节点上限
- ✅ 评估得分应该 > 0（至少不应该是负数）
- ✅ 状态覆盖度应该较好

**训练过程**：
- ✅ 第2次迭代不会崩溃
- ✅ 决策树复杂度逐渐增加
- ✅ 评估得分逐渐提高

**最终结果**：
- 决策树节点数：30-80个（合理范围）
- 神经网络 vs 决策树平局率：> 60%
- 不应该出现先后手100% vs 0%的极端情况

## 如果还有问题

### 如果评估得分仍然很低（< 0）

可能是Oracle本身的问题，测试Oracle：
```bash
python -c "
from stable_baselines3 import DQN
import gymnasium as gym
import gym_env

env = gym.make('TicTacToe-v0', opponent_type='minmax')
model = DQN.load('log/oracle_TicTacToe_selfplay.zip')

wins = 0
for i in range(50):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if done and reward > 0:
            wins += 1
print(f'Oracle vs MinMax: {wins}/50 wins, should be mostly draws')
"
```

### 如果还是有索引错误

检查环境配置：
```bash
python -c "
import gym_env
from train.viper import load_oracle_env
import argparse

args = argparse.Namespace(
    env_name='TicTacToe-v0',
    n_env=8,
    oracle_path='log/oracle_TicTacToe_selfplay.zip',
    seed=42,
    verbose=0,
    tictactoe_opponent='selfplay'
)

env, oracle = load_oracle_env(args)
obs = env.reset()
print(f'Obs shape: {obs.shape}')
print(f'Is vectorized: {len(obs.shape) > 1 and obs.shape[0] > 1}')
action, _ = oracle.predict(obs)
print(f'Action shape: {action.shape if hasattr(action, \"shape\") else \"scalar\"}')
"
```

## 修改的文件清单

1. ✅ `train_viper_improved.py` - 修复向量化环境处理
2. ✅ `QUICK_COMMANDS.md` - 更新推荐配置
3. ✅ `run_improved_viper.sh` - 更新默认参数
4. ✅ `FIX_SUMMARY.md` - 本文档

## 关键代码改动

### 改动1: 提前检测向量化环境
```python
# 在循环开始前
is_vectorized = len(obs.shape) > 1 and obs.shape[0] > 1
n_envs = obs.shape[0] if is_vectorized else 1
```

### 改动2: 决策树预测处理向量化
```python
if isinstance(policy, DecisionTreeClassifier):
    if is_vectorized:
        # 向量化环境：为每个环境预测动作
        action = policy.predict(obs)  # 直接预测整个batch
    else:
        # 单个环境
        action = policy.predict(obs.reshape(1, -1))[0]
```

### 改动3: 随机探索支持向量化
```python
if rand < epsilon_random:
    if is_vectorized:
        action = np.array([np.random.randint(0, env.action_space.n) for _ in range(n_envs)])
    else:
        action = np.random.randint(0, env.action_space.n)
```

所有修复已完成，请重新上传并运行！
