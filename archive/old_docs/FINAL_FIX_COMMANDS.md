# 最终修复命令

## 问题：决策树作为O玩家时非法移动

症状：
- 决策树作为X(先手)：0次非法移动 ✓
- 决策树作为O(后手)：40次非法移动 ❌

## 原因
视角转换代码可能没有生效，原因可能是：
1. 文件没有正确上传
2. Python缓存了旧代码
3. 模型是TreeWrapper，需要额外处理

## 修复步骤

### 步骤1: 确认本地文件是否修复

**在本地执行**：
```bash
cd /Users/adrin/Projects/NOGGINGS/VIPER/viper-verifiable-rl-impl

# 检查battle_nn_vs_tree.py是否有player_id参数
grep -A 5 "def predict(self, obs" battle_nn_vs_tree.py | grep player_id

# 应该看到：def predict(self, obs, player_id=1):
```

### 步骤2: 强制上传并清除缓存

**在本地执行**：
```bash
# 上传修复文件
scp battle_nn_vs_tree.py test_tree_perspective.py user@server:~/viper/
```

**在服务器上执行**：
```bash
ssh user@server
cd ~/viper
conda activate your_env

# 1. 清除Python缓存
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete

# 2. 验证文件是否正确
grep -A 5 "def predict(self, obs" battle_nn_vs_tree.py | grep player_id

# 应该看到：def predict(self, obs, player_id=1):
```

### 步骤3: 测试视角转换

```bash
# 在服务器上运行测试
python test_tree_perspective.py
```

**预期输出**：
```
✓ 不翻转视角会导致非法动作（预期行为）
✓ 翻转视角后可以得到合法动作（正确！）
```

### 步骤4: 重新对战测试

```bash
# 小规模测试（带调试信息）
python battle_nn_vs_tree.py \
  --oracle-path log/oracle_TicTacToe_selfplay.zip \
  --viper-path log/viper_TicTacToe-v0_100_15.joblib \
  --mode both \
  --n-games 10 \
  --debug

# 如果看起来正常，完整测试
python battle_nn_vs_tree.py \
  --oracle-path log/oracle_TicTacToe_selfplay.zip \
  --viper-path log/viper_TicTacToe-v0_100_15.joblib \
  --mode both \
  --n-games 200
```

## 如果还是不行

### 方案A: 检查是否调用了正确的predict

在服务器上创建调试脚本：

```bash
cat > debug_battle.py << 'EOF'
import numpy as np
import joblib
from battle_nn_vs_tree import DecisionTreePlayer

# 加载决策树
player = DecisionTreePlayer("log/viper_TicTacToe-v0_100_15.joblib", debug=True)

# 测试棋盘
board = np.array([1, 0, 0, 0, -1, 0, 0, 0, 0], dtype=np.float32)

print("测试1: X玩家 (player_id=1)")
action1 = player.predict(board, player_id=1)
print(f"动作: {action1}\n")

print("测试2: O玩家 (player_id=-1)")
action2 = player.predict(board, player_id=-1)
print(f"动作: {action2}\n")

print(f"合法动作: {np.where(board == 0)[0]}")
EOF

python debug_battle.py
```

### 方案B: 直接修改保存的模型类

如果问题持续，创建一个包装类：

```bash
cat > fix_model_wrapper.py << 'EOF'
import joblib
import numpy as np

# 加载原始模型
model = joblib.load("log/viper_TicTacToe-v0_100_15.joblib")

# 创建支持视角转换的包装类
class PerspectiveAwareTree:
    def __init__(self, tree):
        self.tree = tree if not hasattr(tree, 'tree') else tree.tree

    def predict(self, obs):
        """标准predict接口（用于战斗）"""
        return self.tree.predict(obs)

    def predict_with_perspective(self, obs, player_id=1):
        """带视角转换的predict"""
        if player_id == -1:
            obs = -obs
        return self.tree.predict(obs.reshape(1, -1))[0]

# 保存新模型
wrapper = PerspectiveAwareTree(model)
joblib.dump(wrapper, "log/viper_TicTacToe-v0_100_15_fixed.joblib")
print("已创建修复后的模型: log/viper_TicTacToe-v0_100_15_fixed.joblib")
EOF

python fix_model_wrapper.py
```

## 最坏情况：重新训练

如果所有修复都失败，可能是训练数据本身有问题。重新训练时加入视角增强：

```bash
python train_viper_improved.py \
  --env-name TicTacToe-v0 \
  --oracle-path log/oracle_TicTacToe_selfplay.zip \
  --n-iter 60 \
  --total-timesteps 100000 \
  --max-leaves 100 \
  --max-depth 15 \
  --use-augmentation \
  --exploration-strategy decay \
  --verbose 1
```

注意：`--use-augmentation` 会使用对称性增强，这应该包含视角翻转。

## 验证成功的标准

```
神经网络 vs 决策树：
  平局率: > 40% ✓
  非法移动: 0 ✓

决策树 vs 神经网络：
  平局率: > 40% ✓
  非法移动: 0 ✓
```

## 调试检查清单

- [ ] 本地文件有 `def predict(self, obs, player_id=1)`
- [ ] 文件已上传到服务器
- [ ] 服务器上的文件包含 `player_id` 参数
- [ ] 清除了 `__pycache__` 目录
- [ ] `test_tree_perspective.py` 显示翻转后可以得到合法动作
- [ ] 对战时调用了 `predict(obs, player_id=current_player_id)`

按照这个清单逐项检查！
