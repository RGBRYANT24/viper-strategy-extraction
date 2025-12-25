"""
测试MinMax的开局是否确定性
"""
import numpy as np
from gym_env.tictactoe import TicTacToeEnv

# 创建MinMax环境
env = TicTacToeEnv(opponent_type='minmax', minmax_depth=9)

print("=" * 70)
print("测试MinMax开局")
print("=" * 70)

# 测试10次空棋盘，看MinMax第一步
minmax_first_moves = []
for i in range(10):
    env.board = np.zeros(9, dtype=np.float32)
    action = env._opponent_move()
    minmax_first_moves.append(action)
    print(f"测试 {i+1}: MinMax第一步选择位置 {action}")

unique = np.unique(minmax_first_moves)
print(f"\nMinMax第一步的唯一值: {unique}")
if len(unique) == 1:
    print(f"⚠️  MinMax总是选位置{unique[0]}（确定性算法导致！）")

# 测试：如果玩家先走不同位置，MinMax如何反应
print("\n" + "=" * 70)
print("测试：玩家占据不同位置后，MinMax的反应")
print("=" * 70)

for player_pos in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
    env.board = np.zeros(9, dtype=np.float32)
    env.board[player_pos] = 1
    action = env._opponent_move()
    print(f"玩家在位置{player_pos} → MinMax选位置{action}")
