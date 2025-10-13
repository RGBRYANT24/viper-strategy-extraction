# 🚨 紧急修复：决策树视角问题

## 问题症状

```
决策树 (O) 获胜: 0 局 (0.0%)
非法移动总数: 200
  - 决策树 非法移动: 200
```

决策树作为后手时，所有动作都是非法的！

## 根本原因

**视角不匹配问题**：

1. **训练时**：决策树总是从X玩家视角学习
   - 自己的棋子 = 1
   - 对手的棋子 = -1
   - 空位 = 0

2. **对战时**：决策树作为O玩家（后手）
   - 环境给的棋盘：X=1, O=-1
   - 但决策树认为自己应该是1！
   - 结果：决策树看到的棋盘完全反了

## 修复方法

### 在本地上传修复文件

```bash
# 上传修复后的battle_nn_vs_tree.py
scp /Users/adrin/Projects/NOGGINGS/VIPER/viper-verifiable-rl-impl/battle_nn_vs_tree.py user@server:~/viper/
```

### 在服务器上重新测试

```bash
ssh user@server
cd ~/viper
conda activate your_env

# 重新测试
python battle_nn_vs_tree.py \
  --oracle-path log/oracle_TicTacToe_selfplay.zip \
  --viper-path log/viper_TicTacToe-v0_100_15.joblib \
  --mode both \
  --n-games 200
```

## 修复内容

### 1. DecisionTreePlayer.predict() 添加视角转换

```python
def predict(self, obs, player_id=1):
    """
    Args:
        player_id: 1=X玩家, -1=O玩家
    """
    # 如果是O玩家，翻转棋盘视角
    if player_id == -1:
        obs_transformed = -obs  # X变-1, O变1
    else:
        obs_transformed = obs

    action = self.model.predict(obs_transformed.reshape(1, -1))[0]
    return action
```

### 2. battle_two_players() 传递player_id

```python
# 预测动作时传入当前玩家ID
if isinstance(current_agent, DecisionTreePlayer):
    action = current_agent.predict(obs, player_id=current_player_id)
else:
    action = current_agent.predict(obs)
```

## 预期结果

修复后应该看到：

```
神经网络 vs 决策树：
  平局率: 50-80% ✓
  非法移动: 0 ✓

决策树 vs 神经网络：
  平局率: 50-80% ✓
  非法移动: 0 ✓
```

## 为什么会有这个问题？

这是VIPER训练的一个常见陷阱：

1. **训练时**：Oracle(神经网络)总是扮演X玩家对抗环境中的O玩家
2. **VIPER学习**：决策树学习Oracle的策略，也是从X的视角
3. **对战时**：决策树可能需要扮演O玩家，但它不知道如何转换视角！

## 如何避免？

### 方法1：训练时使用双视角数据（推荐）

在VIPER训练时，同时收集X和O视角的数据：

```python
# 训练数据包含两种视角
trajectory.append((obs, action, weight))           # X视角
trajectory.append((-obs, action, weight))          # O视角（翻转）
```

### 方法2：对战时动态转换（当前方案）

在对战时检测玩家角色并转换棋盘视角（已实现）

## 验证修复

```bash
# 测试1: 基础对战（应该没有非法移动）
python battle_nn_vs_tree.py \
  --oracle-path log/oracle_TicTacToe_selfplay.zip \
  --viper-path log/viper_TicTacToe-v0_100_15.joblib \
  --mode both \
  --n-games 50

# 测试2: 详细调试（查看前5步）
python battle_nn_vs_tree.py \
  --oracle-path log/oracle_TicTacToe_selfplay.zip \
  --viper-path log/viper_TicTacToe-v0_100_15.joblib \
  --mode both \
  --n-games 5 \
  --verbose \
  --debug

# 测试3: 完整评估（200局）
python battle_nn_vs_tree.py \
  --oracle-path log/oracle_TicTacToe_selfplay.zip \
  --viper-path log/viper_TicTacToe-v0_100_15.joblib \
  --mode all \
  --n-games 200
```

立即上传修复文件并重新测试！
