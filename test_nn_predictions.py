"""
测试神经网络的预测是否正常
"""
import numpy as np
from stable_baselines3 import DQN

# 加载神经网络
nn_path = "log/oracle_TicTacToe_selfplay.zip"


model = DQN.load(nn_path)

print("=" * 70)
print("测试神经网络预测")
print("=" * 70)

print('nn_path', nn_path)

test_cases = [
    ("空棋盘", np.zeros(9, dtype=np.float32)),
    ("X在位置0", np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)),
    ("X在位置1", np.array([0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)),
    ("X在位置4(中心)", np.array([0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)),
    ("X在0,O在1", np.array([1, -1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)),
    ("X在0,O在4", np.array([1, 0, 0, 0, -1, 0, 0, 0, 0], dtype=np.float32)),
]

print("\n逐个状态测试:")
for name, state in test_cases:
    action, _ = model.predict(state, deterministic=True)
    legal_actions = np.where(state == 0)[0]

    print(f"\n{name}:")
    print(f"  状态: {state}")
    print(f"  预测动作: {action}")
    print(f"  合法动作: {legal_actions}")
    print(f"  是否合法: {action in legal_actions}")

    if action not in legal_actions:
        print(f"  ⚠️  非法动作！")

# 批量测试：模拟对战开局
print("\n" + "=" * 70)
print("模拟100个开局（神经网络先手）")
print("=" * 70)

np.random.seed(42)
first_moves = []

for _ in range(100):
    state = np.zeros(9, dtype=np.float32)
    action, _ = model.predict(state, deterministic=True)
    first_moves.append(action)

unique, counts = np.unique(first_moves, return_counts=True)
print("\n神经网络第一步分布:")
for val, count in zip(unique, counts):
    print(f"  位置{val}: {count}次 ({count/100*100:.1f}%)")

if len(unique) == 1:
    print("\n⚠️  警告：神经网络开局总是走同一个位置！")
    print("这可能导致VIPER训练数据不平衡。")

# 测试：让对手先走后，神经网络的反应
print("\n" + "=" * 70)
print("测试：对手占据不同位置后，神经网络的反应")
print("=" * 70)

for opp_pos in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
    state = np.zeros(9, dtype=np.float32)
    state[opp_pos] = -1  # 对手占据opp_pos
    action, _ = model.predict(state, deterministic=True)
    legal = action in np.where(state == 0)[0]

    status = "✓" if legal else "✗"
    print(f"  对手在位置{opp_pos} → 神经网络选{action} {status}")

# 获取Q值分布
print("\n" + "=" * 70)
print("Q值分析（空棋盘）")
print("=" * 70)

import torch
state = np.zeros(9, dtype=np.float32)
state_tensor = torch.FloatTensor(state).unsqueeze(0).to(model.device)

with torch.no_grad():
    q_values = model.q_net(state_tensor).cpu().numpy()[0]

print("\n每个动作的Q值:")
for i, q in enumerate(q_values):
    print(f"  位置{i}: Q={q:.4f}")

print(f"\n最佳动作: {np.argmax(q_values)}")
print(f"Q值范围: [{q_values.min():.4f}, {q_values.max():.4f}]")
print(f"Q值标准差: {q_values.std():.4f}")

if q_values.std() < 0.01:
    print("\n⚠️  警告：Q值几乎相同，神经网络可能没有学到有效策略！")
