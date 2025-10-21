"""
测试"奇偶性假设"：
神经网络是否在奇数个棋子的局面下表现差？
"""
import numpy as np
import torch
from stable_baselines3 import DQN

# 加载模型
print("加载神经网络模型...")
model = DQN.load("log/oracle_TicTacToe_selfplay.zip")
device = next(model.q_net.parameters()).device
print(f"模型设备: {device}")

def get_q_values(board, flip=False):
    """获取Q值"""
    obs = board.copy()
    if flip:
        obs = -obs

    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model.q_net(obs_tensor).cpu().numpy()[0]
    return q_values

def analyze_position(board, description, flip=False):
    """分析一个局面"""
    num_pieces = np.sum(board != 0)
    num_x = np.sum(board == 1)
    num_o = np.sum(board == -1)

    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"棋子总数: {num_pieces} (X:{num_x}, O:{num_o})")
    print(f"视角翻转: {'是' if flip else '否'}")
    print(f"棋盘:")

    symbols = {1: 'X', -1: 'O', 0: '.'}
    for i in range(3):
        row = " | ".join([symbols[int(board[i*3 + j])] for j in range(3)])
        print(f"  {row}")

    q_values = get_q_values(board, flip)
    legal_actions = np.where(board == 0)[0]

    print(f"\nQ值分析:")
    print(f"合法动作: {legal_actions}")

    # 显示所有Q值
    print(f"\n所有位置Q值:")
    for i in range(3):
        row_q = [f"{q_values[i*3+j]:7.3f}" for j in range(3)]
        row_status = []
        for j in range(3):
            pos = i*3 + j
            if pos in legal_actions:
                row_status.append("合法")
            else:
                row_status.append("占用")
        print(f"  {' | '.join(row_q)}  ({', '.join(row_status)})")

    # 最佳动作
    if len(legal_actions) > 0:
        masked_q = np.full(9, -np.inf)
        masked_q[legal_actions] = q_values[legal_actions]
        best_action = np.argmax(masked_q)
        best_q = q_values[best_action]

        print(f"\n推荐动作: {best_action} (Q值 = {best_q:.3f})")

        # 显示合法动作的Q值排序
        legal_q = [(a, q_values[a]) for a in legal_actions]
        legal_q.sort(key=lambda x: x[1], reverse=True)
        print(f"\n合法动作Q值排序:")
        for rank, (action, q) in enumerate(legal_q[:5], 1):
            print(f"  {rank}. 位置{action}: Q={q:.3f}")

print("\n" + "="*70)
print("测试假设：神经网络在奇数棋子局面下表现差")
print("="*70)

# 测试1：偶数棋子（训练时见过的类型）
print("\n\n## 测试组1：偶数棋子局面（训练时常见）##")

# 2个棋子：X先走中心，O走角
board_2 = np.zeros(9, dtype=np.float32)
board_2[4] = 1  # X在中心
board_2[0] = -1  # O在角
analyze_position(board_2, "2个棋子：X中心，O角 - 从X视角", flip=False)

# 4个棋子
board_4 = np.zeros(9, dtype=np.float32)
board_4[4] = 1   # X中心
board_4[0] = -1  # O左上
board_4[8] = 1   # X右下
board_4[2] = -1  # O右上
analyze_position(board_4, "4个棋子 - 从X视角", flip=False)

print("\n\n## 测试组2：奇数棋子局面（训练时罕见/不存在）##")

# 1个棋子：对手先走中心，我后手
board_1 = np.zeros(9, dtype=np.float32)
board_1[4] = 1  # 对手(X)在中心
analyze_position(board_1, "1个棋子：对手走中心，我后手 - 需要翻转视角", flip=True)

# 1个棋子：对手先走角
board_1_corner = np.zeros(9, dtype=np.float32)
board_1_corner[0] = 1  # 对手(X)在角
analyze_position(board_1_corner, "1个棋子：对手走角，我后手 - 需要翻转视角", flip=True)

# 3个棋子：我后手
board_3 = np.zeros(9, dtype=np.float32)
board_3[0] = 1   # 对手左上
board_3[4] = -1  # 我中心
board_3[8] = 1   # 对手右下
analyze_position(board_3, "3个棋子：我后手 - 需要翻转视角", flip=True)

# 5个棋子：我后手
board_5 = np.zeros(9, dtype=np.float32)
board_5[0] = 1   # 对手
board_5[4] = -1  # 我
board_5[8] = 1   # 对手
board_5[1] = -1  # 我
board_5[7] = 1   # 对手
analyze_position(board_5, "5个棋子：我后手 - 需要翻转视角", flip=True)

print("\n\n" + "="*70)
print("分析结论")
print("="*70)
print("观察以下指标来验证假设：")
print("1. Q值的绝对大小：奇数棋子局面的Q值是否普遍较低/不稳定？")
print("2. Q值的区分度：奇数棋子局面下，好动作和坏动作的Q值差异是否很小？")
print("3. 推荐动作的合理性：奇数棋子局面下的推荐动作是否明显不合理？")
print("="*70)
