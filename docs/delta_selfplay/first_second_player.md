# 先手/后手训练机制详解

## 问题：神经网络如何学习后手(O)操作？

**答案**: 通过 **视角翻转 (Perspective Flipping)** 技术，让神经网络看到统一的输入表示。

---

## 核心代码位置

### 1. 随机决定先手/后手 (50%概率)

📍 **文件**: `gym_env/tictactoe_delta_selfplay.py` 第114行

```python
def reset(self, seed=None, options=None):
    # ⭐ 关键: 随机决定先手/后手
    self.play_as_o = (np.random.random() < self.play_as_o_prob)
    #                                      ^^^^^^^^^^^^^^^^^^^
    #                                      默认 0.5 = 50% 后手

    # 如果是后手(O)，对手先走
    if self.play_as_o:
        opponent_action = self._opponent_move()
        self.board[opponent_action] = 1  # 对手下X
```

### 2. 视角翻转机制 (核心)

📍 **文件**: `gym_env/tictactoe_delta_selfplay.py` 第182-199行

```python
def _get_observation(self):
    """
    获取观察（从自己的视角）

    关键：无论先手后手，神经网络输入必须一致
    - 自己的棋子总是表示为 1
    - 对手的棋子总是表示为 -1
    """
    if self.play_as_o:
        # ⭐ 后手O: 翻转棋盘视角
        # 实际棋盘: X=1, O=-1
        # 网络输入: 自己(O)=1, 对手(X)=-1
        return -self.board.copy()  # 关键：取负号！
    else:
        # 先手X: 直接返回
        return self.board.copy()
```

---

## 工作原理图解

### 先手 (X) 训练

```
实际棋盘:          神经网络输入:
  X | . | O          1 | 0 | -1
 -----------        -----------
  . | . | .          0 | 0 | 0
 -----------        -----------
  . | . | .          0 | 0 | 0

自己=X=1           自己=1 ✓
对手=O=-1          对手=-1 ✓
```

### 后手 (O) 训练

```
实际棋盘:          神经网络输入 (翻转后):
  X | . | O          -1 | 0 | 1
 -----------        -----------
  . | . | .          0  | 0 | 0
 -----------        -----------
  . | . | .          0  | 0 | 0

自己=O=-1          自己=1 ✓  (−(−1)=1)
对手=X=1           对手=-1 ✓  (−1=−1)
```

**关键洞察**: 无论先手后手，神经网络**总是看到"自己=1, 对手=-1"**！

---

## 完整训练流程示例

### Episode A: 后手训练

```python
# 1. reset()
play_as_o = True  # 50%概率

# 2. 对手(X)先走中心
board = [0, 0, 0, 0, 1, 0, 0, 0, 0]

# 3. 神经网络看到 (翻转)
obs = -board = [0, 0, 0, 0, -1, 0, 0, 0, 0]
#                           ↑对手=-1

# 4. 神经网络预测
action = 0  # 下角落

# 5. 执行 (我方O)
board[0] = -1  # [−1, 0, 0, 0, 1, ...]

# 6. 继续对战...
# 7. 获得奖励，学习
```

### Episode B: 先手训练

```python
# 1. reset()
play_as_o = False  # 50%概率

# 2. 神经网络先走
obs = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # 不翻转

# 3. 神经网络预测
action = 4  # 下中心

# 4. 执行 (我方X)
board[4] = 1  # [0, 0, 0, 0, 1, ...]

# 5. 对手下棋...
# 6. 获得奖励，学习
```

---

## 为什么需要翻转？

### ❌ 不翻转的问题

```python
# 先手训练
输入: [1, 0, -1, ...]  # 自己X=1
学到: "当我=1时，下这里"

# 后手推理 (不翻转)
输入: [-1, 0, 1, ...]  # 自己O=-1 ❌
问题: 从未见过"自己=-1"！
结果: 完全失效
```

### ✅ 翻转的好处

```python
# 先手训练
输入: [1, 0, -1, ...]  # 自己=1, 对手=-1

# 后手推理 (翻转)
输入: [1, 0, -1, ...]  # 翻转后: 自己=1, 对手=-1 ✓
结果: 和先手一样！用同一策略
```

---

## 验证代码

```python
from gym_env.tictactoe_delta_selfplay import TicTacToeDeltaSelfPlayEnv
from gym_env.policies import RandomPlayerPolicy
from gymnasium import spaces
import numpy as np

# 创建环境
obs_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.float32)
act_space = spaces.Discrete(9)
baseline_pool = [RandomPlayerPolicy(obs_space, act_space)]

env = TicTacToeDeltaSelfPlayEnv(
    baseline_pool=baseline_pool,
    play_as_o_prob=1.0  # 强制后手
)

# 测试
obs, _ = env.reset(seed=42)

print(f"角色: {'O后手' if env.play_as_o else 'X先手'}")
print(f"\n实际棋盘:\n{env.board.reshape(3, 3)}")
print(f"\n网络输入:\n{obs.reshape(3, 3)}")

# 验证
if env.play_as_o:
    assert np.allclose(obs, -env.board)
    print("\n✓ 翻转正确！")
```

---

## 总结

| 组件 | 代码位置 | 作用 |
|------|---------|------|
| 先后手决定 | `reset()` L114 | 50%概率后手 |
| 视角翻转 | `_get_observation()` L190-194 | 后手时取负 |
| 标记处理 | `step()` L147 | 正确设置X/O |

**核心**: 通过翻转，神经网络用**一个策略**同时学会先手和后手！🎯
