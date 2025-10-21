"""
分析Q值分布，验证是否存在"过度惩罚"或"策略固化"问题
"""
import numpy as np
import torch
from stable_baselines3 import DQN

# 加载模型
print("加载神经网络模型...")
model = DQN.load("log/oracle_TicTacToe_selfplay.zip")
device = next(model.q_net.parameters()).device

def get_q_values(board):
    """获取Q值"""
    obs_tensor = torch.FloatTensor(board).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model.q_net(obs_tensor).cpu().numpy()[0]
    return q_values

print("\n" + "="*70)
print("分析1：非法移动惩罚的影响")
print("="*70)

# 测试不同占用情况下的Q值
boards = []

# 空棋盘（先手）
empty = np.zeros(9, dtype=np.float32)
boards.append(("空棋盘（先手）", empty, False))

# 对手走了1步（后手）
one_piece = np.zeros(9, dtype=np.float32)
one_piece[0] = 1
boards.append(("对手走角（后手）", one_piece, True))

# 对手走了1步 - 中心
one_center = np.zeros(9, dtype=np.float32)
one_center[4] = 1
boards.append(("对手走中心（后手）", one_center, True))

# 复杂局面
complex_board = np.zeros(9, dtype=np.float32)
complex_board[0] = 1
complex_board[1] = -1
complex_board[4] = 1
complex_board[8] = -1
boards.append(("复杂局面（4子）", complex_board, False))

all_legal_q = []
all_illegal_q = []

for desc, board, flip in boards:
    obs = -board if flip else board
    q_values = get_q_values(obs)

    legal_mask = (board == 0)
    illegal_mask = (board != 0)

    legal_q = q_values[legal_mask]
    illegal_q = q_values[illegal_mask]

    all_legal_q.extend(legal_q)
    all_illegal_q.extend(illegal_q)

    print(f"\n{desc}:")
    print(f"  合法位置Q值: 均值={legal_q.mean():.3f}, 标准差={legal_q.std():.3f}, "
          f"范围=[{legal_q.min():.3f}, {legal_q.max():.3f}]")
    if len(illegal_q) > 0:
        print(f"  非法位置Q值: 均值={illegal_q.mean():.3f}, 标准差={illegal_q.std():.3f}, "
              f"范围=[{illegal_q.min():.3f}, {illegal_q.max():.3f}]")
    if len(illegal_q) > 0:
        print(f"  合法/非法Q值差距: {legal_q.mean() - illegal_q.mean():.3f}")
    else:
        print(f"  合法/非法Q值差距: N/A (没有非法位置)")

print(f"\n{'='*70}")
print("总体统计:")
all_legal_q = np.array(all_legal_q)
all_illegal_q = np.array(all_illegal_q)

print(f"所有合法位置Q值: 均值={all_legal_q.mean():.3f}, 标准差={all_legal_q.std():.3f}")
print(f"所有非法位置Q值: 均值={all_illegal_q.mean():.3f}, 标准差={all_illegal_q.std():.3f}")
print(f"差距: {all_legal_q.mean() - all_illegal_q.mean():.3f}")

print("\n" + "="*70)
print("分析2：策略多样性测试")
print("="*70)

# 测试从同一局面出发，不同初始状态下的决策是否相同
test_positions = []

# 创建几个"对手走角"的场景
for corner in [0, 2, 6, 8]:
    board = np.zeros(9, dtype=np.float32)
    board[corner] = 1
    test_positions.append((f"对手走角{corner}", board))

print("\n对手走不同角时，神经网络的应对：")
print("(理论上都应该走中心，位置4)")

responses = []
for desc, board in test_positions:
    obs = -board
    q_values = get_q_values(obs)

    legal_mask = (board == 0)
    masked_q = np.full(9, -np.inf)
    masked_q[legal_mask] = q_values[legal_mask]

    best_action = np.argmax(masked_q)
    best_q = q_values[best_action]

    responses.append(best_action)

    # 检查位置4的排名
    legal_actions = np.where(legal_mask)[0]
    legal_q_pairs = [(a, q_values[a]) for a in legal_actions]
    legal_q_pairs.sort(key=lambda x: x[1], reverse=True)

    rank_4 = None
    for rank, (action, q) in enumerate(legal_q_pairs, 1):
        if action == 4:
            rank_4 = rank
            break

    marker = "✓" if best_action == 4 else "✗"
    print(f"  {desc}: 推荐动作={best_action} (Q={best_q:.3f}) {marker}")
    print(f"    位置4的Q值: {q_values[4]:.3f}, 排名: {rank_4}")

# 检查策略一致性
if len(set(responses)) == 1:
    print(f"\n策略一致性: ✓ 所有场景都选择相同动作 ({responses[0]})")
else:
    print(f"\n策略一致性: ✗ 不同场景选择不同动作: {set(responses)}")

if all(r == 4 for r in responses):
    print("结论: ✓ 策略正确！所有场景都走中心")
else:
    print("结论: ✗ 策略错误！没有正确识别'对手走角→我走中心'的最优策略")

print("\n" + "="*70)
print("分析3：决策边界分析")
print("="*70)

# 对于"对手走角"场景，比较位置4（正确）和位置7（实际选择）的Q值
board = np.zeros(9, dtype=np.float32)
board[0] = 1
obs = -board
q_values = get_q_values(obs)

print("\n场景：对手走左上角")
print(f"位置4（中心，最优）: Q = {q_values[4]:.4f}")
print(f"位置7（左下中间，神经网络实际选择）: Q = {q_values[7]:.4f}")
print(f"Q值差距: {q_values[7] - q_values[4]:.4f}")

# 如果Q值差距很小，说明网络没有学到明确的策略
if abs(q_values[7] - q_values[4]) < 0.5:
    print("\n⚠️  Q值差距太小！网络对最优动作的判断不够确定")
    print("   可能原因：")
    print("   1. Self-play训练不充分，没有遇到足够多的'对手走角'场景")
    print("   2. 对手太弱，导致'随便走都能赢'，学不到精细策略")
    print("   3. 奖励信号太稀疏（只有输赢，没有中间评分）")
else:
    print("\nQ值差距合理，但策略错误！")
    print("   说明网络学到了错误的策略")

print("\n" + "="*70)
print("总结诊断")
print("="*70)
print("\n根据以上分析，神经网络后手表现差的主要原因是:")
print("\n1. ✗ 非法移动惩罚过度？")
print("   答案: 不是主要原因。非法位置Q值确实很低（-9~-11），")
print("   但这是正常的，不会影响合法动作之间的相对顺序。")

print("\n2. ✓ Self-play训练不到位？")
print("   答案: 很可能！网络没有学到'对手走角→我走中心'这样的基本策略。")
print("   Self-play的问题：")
print("   - 如果对手也很弱，双方都在犯错，学不到最优策略")
print("   - 缺少和MinMax这样的'完美老师'对战")
print("   - 可能陷入某种次优的稳定策略（局部最优）")

print("\n3. ✓ 策略固化？")
print("   答案: 有可能。检查上面'策略多样性'的结果。")
print("   如果网络对不同的'对手走角'场景给出不同的（都错误的）应对，")
print("   说明策略是混乱的而不是固化的。")

print("\n推荐解决方案：")
print("1. 用MinMax作为对手重新训练（或混合训练：部分MinMax，部分self-play）")
print("2. 增加训练轮数")
print("3. 调整奖励函数：不只是输赢（-1/0/+1），而是考虑棋局质量")
print("="*70)
