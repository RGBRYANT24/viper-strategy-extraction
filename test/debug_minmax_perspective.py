#!/usr/bin/env python3
"""
调试 MinMax 算法的玩家视角问题
测试 player=1 和 player=-1 时，minimax 的行为是否正确
"""

import numpy as np
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gym_env.policies.baseline_policies import MinMaxPlayerPolicy
from gymnasium import spaces


def print_board(board, title=""):
    """美化打印棋盘"""
    if title:
        print(f"\n{title}")
    symbols = {1: 'X', -1: 'O', 0: '.'}
    board_2d = board.reshape(3, 3)
    print()
    for i in range(3):
        row = " | ".join([symbols[int(cell)] for cell in board_2d[i]])
        print(f"  {row}")
        if i < 2:
            print(" -----------")
    print()


def test_scenario_1():
    """
    测试场景1：X (player=1) 即将获胜
    棋盘:
    X | X | .
    ---------
    . | . | .
    ---------
    . | . | .

    期望：
    - 如果从 X (1) 视角：应该选择位置 2 获胜
    - 如果从 O (-1) 视角：应该选择位置 2 阻止X获胜
    """
    print("=" * 80)
    print("测试场景1: X即将获胜 (位置0和1已有X)")
    print("=" * 80)

    board = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    print_board(board, "当前棋盘:")

    # 创建策略
    obs_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.float32)
    act_space = spaces.Discrete(9)
    policy = MinMaxPlayerPolicy(obs_space, act_space)

    # 测试从 X (player=1) 的视角
    print("🔍 测试1: 从 X (player=1) 的视角")
    action_as_x = policy._minmax_move(board.copy(), player=1)
    print(f"   选择的动作: {action_as_x}")
    print(f"   预期动作: 2 (获胜)")

    # 模拟这个动作
    test_board = board.copy()
    test_board[action_as_x] = 1
    print_board(test_board, "   X下在位置{action_as_x}后:")

    result_x = action_as_x == 2
    print(f"   结果: {'✓ 正确' if result_x else '✗ 错误'}")
    print()

    # 测试从 O (player=-1) 的视角
    print("🔍 测试2: 从 O (player=-1) 的视角")
    action_as_o = policy._minmax_move(board.copy(), player=-1)
    print(f"   选择的动作: {action_as_o}")
    print(f"   预期动作: 2 (阻止X获胜)")

    # 模拟这个动作
    test_board = board.copy()
    test_board[action_as_o] = -1
    print_board(test_board, f"   O下在位置{action_as_o}后:")

    result_o = action_as_o == 2
    print(f"   结果: {'✓ 正确' if result_o else '✗ 错误'}")
    print()

    return result_x and result_o


def test_scenario_2():
    """
    测试场景2：O (player=-1) 即将获胜
    棋盘:
    O | . | X
    ---------
    O | . | X
    ---------
    . | . | .

    期望：
    - 如果从 X (1) 视角：应该选择位置 6 阻止O获胜
    - 如果从 O (-1) 视角：应该选择位置 6 获胜
    """
    print("=" * 80)
    print("测试场景2: O即将获胜 (位置0和3已有O)")
    print("=" * 80)

    board = np.array([-1, 0, 1, -1, 0, 1, 0, 0, 0], dtype=np.float32)
    print_board(board, "当前棋盘:")

    # 创建策略
    obs_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.float32)
    act_space = spaces.Discrete(9)
    policy = MinMaxPlayerPolicy(obs_space, act_space)

    # 测试从 X (player=1) 的视角
    print("🔍 测试1: 从 X (player=1) 的视角")
    action_as_x = policy._minmax_move(board.copy(), player=1)
    print(f"   选择的动作: {action_as_x}")
    print(f"   预期动作: 6 (阻止O获胜)")

    test_board = board.copy()
    test_board[action_as_x] = 1
    print_board(test_board, f"   X下在位置{action_as_x}后:")

    result_x = action_as_x == 6
    print(f"   结果: {'✓ 正确' if result_x else '✗ 错误'}")
    print()

    # 测试从 O (player=-1) 的视角
    print("🔍 测试2: 从 O (player=-1) 的视角")
    action_as_o = policy._minmax_move(board.copy(), player=-1)
    print(f"   选择的动作: {action_as_o}")
    print(f"   预期动作: 6 (获胜)")

    test_board = board.copy()
    test_board[action_as_o] = -1
    print_board(test_board, f"   O下在位置{action_as_o}后:")

    result_o = action_as_o == 6
    print(f"   结果: {'✓ 正确' if result_o else '✗ 错误'}")
    print()

    return result_x and result_o


def test_scenario_3():
    """
    测试场景3：复杂局面
    棋盘:
    X | O | X
    ---------
    . | O | .
    ---------
    . | . | .

    O (player=-1) 可以在位置7获胜 (完成1-4-7连线)

    期望：
    - 如果从 O (-1) 视角：应该选择位置 7 获胜
    - 如果从 X (1) 视角：应该选择位置 7 阻止O获胜
    """
    print("=" * 80)
    print("测试场景3: O可以通过位置7获胜 (1-4-7连线)")
    print("=" * 80)

    board = np.array([1, -1, 1, 0, -1, 0, 0, 0, 0], dtype=np.float32)
    print_board(board, "当前棋盘:")
    print("   注意：O已经占据位置1和4，如果下在7就能连成一线")
    print()

    # 创建策略
    obs_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.float32)
    act_space = spaces.Discrete(9)
    policy = MinMaxPlayerPolicy(obs_space, act_space)

    # 测试从 O (player=-1) 的视角
    print("🔍 测试1: 从 O (player=-1) 的视角")
    action_as_o = policy._minmax_move(board.copy(), player=-1)
    print(f"   选择的动作: {action_as_o}")
    print(f"   预期动作: 7 (获胜)")

    test_board = board.copy()
    test_board[action_as_o] = -1
    print_board(test_board, f"   O下在位置{action_as_o}后:")

    result_o = action_as_o == 7
    print(f"   结果: {'✓ 正确' if result_o else '✗ 错误 - O应该直接获胜！'}")
    print()

    # 测试从 X (player=1) 的视角
    print("🔍 测试2: 从 X (player=1) 的视角")
    action_as_x = policy._minmax_move(board.copy(), player=1)
    print(f"   选择的动作: {action_as_x}")
    print(f"   预期动作: 7 (阻止O获胜)")

    test_board = board.copy()
    test_board[action_as_x] = 1
    print_board(test_board, f"   X下在位置{action_as_x}后:")

    result_x = action_as_x == 7
    print(f"   结果: {'✓ 正确' if result_x else '✗ 错误 - X必须阻止O！'}")
    print()

    return result_o and result_x


def debug_minimax_scoring():
    """
    详细调试 minimax 内部评分过程
    这个函数会打印出每个可能动作的评分，帮助理解算法的决策过程
    """
    print("=" * 80)
    print("调试 MinMax 内部评分过程")
    print("=" * 80)

    # 简单场景：X即将获胜
    board = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    print_board(board, "测试棋盘 (X在位置0和1):")

    obs_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.float32)
    act_space = spaces.Discrete(9)
    policy = MinMaxPlayerPolicy(obs_space, act_space)

    legal_actions = np.where(board == 0)[0]

    print("从 X (player=1) 视角评估每个可能的动作：")
    print("-" * 60)
    for action in legal_actions:
        test_board = board.copy()
        test_board[action] = 1  # X下这步
        score = policy._minimax(test_board, 0, False, player=1, alpha=float('-inf'), beta=float('inf'))
        print(f"  动作 {action}: 分数 = {score:+.1f}", end="")
        if action == 2:
            print("  ← X直接获胜！")
        else:
            print()

    print()
    print("从 O (player=-1) 视角评估每个可能的动作：")
    print("-" * 60)
    for action in legal_actions:
        test_board = board.copy()
        test_board[action] = -1  # O下这步
        score = policy._minimax(test_board, 0, False, player=-1, alpha=float('-inf'), beta=float('inf'))
        print(f"  动作 {action}: 分数 = {score:+.1f}", end="")
        if action == 2:
            print("  ← 阻止X获胜！")
        else:
            print()

    print()
    print("🔍 分析：")
    print("   如果算法正确，两个视角下位置2都应该得到最高分")
    print("   - X视角：位置2直接获胜 (分数应该是 +10)")
    print("   - O视角：位置2阻止失败 (分数应该最高，因为其他位置会输)")
    print()


def test_symmetric_positions():
    """
    测试对称局面下的表现
    检查算法在相同优势程度的情况下是否一致
    """
    print("=" * 80)
    print("测试场景4: 对称局面")
    print("=" * 80)

    # 场景A：X在第一行即将获胜
    board_a = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    print_board(board_a, "场景A: X在第一行即将获胜")

    # 场景B：O在第一行即将获胜（对称）
    board_b = np.array([-1, -1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    print_board(board_b, "场景B: O在第一行即将获胜（对称）")

    obs_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.float32)
    act_space = spaces.Discrete(9)
    policy = MinMaxPlayerPolicy(obs_space, act_space)

    # X在场景A中应该选2（获胜）
    action_a = policy._minmax_move(board_a.copy(), player=1)
    print(f"场景A - X的选择: {action_a} (预期: 2)")

    # O在场景B中应该选2（获胜）
    action_b = policy._minmax_move(board_b.copy(), player=-1)
    print(f"场景B - O的选择: {action_b} (预期: 2)")

    result = (action_a == 2) and (action_b == 2)
    print(f"\n结果: {'✓ 对称局面处理正确' if result else '✗ 对称局面处理有问题'}")
    print()

    return result


if __name__ == "__main__":
    print("\n" + "🔍 " * 40)
    print("MinMax 玩家视角调试工具")
    print("🔍 " * 40 + "\n")

    results = []

    # 运行测试场景
    results.append(("场景1: X即将获胜", test_scenario_1()))

    results.append(("场景2: O即将获胜", test_scenario_2()))

    results.append(("场景3: 复杂局面", test_scenario_3()))

    results.append(("场景4: 对称局面", test_symmetric_positions()))

    # 调试内部评分
    debug_minimax_scoring()

    # 总结
    print("=" * 80)
    print("测试总结")
    print("=" * 80)
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{status}: {name}")

    all_passed = all(result[1] for result in results)

    print()
    if all_passed:
        print("🎉 所有测试通过！MinMax算法的玩家视角处理正确。")
    else:
        print("⚠️  发现问题！MinMax算法在处理不同玩家视角时存在bug。")
        print()
        print("可能的问题原因：")
        print("1. _minimax 函数中的评分逻辑假设 player 总是 +1")
        print("2. 当 player=-1 时，'winner == player' 的逻辑反转了")
        print("3. 建议修改：在评分时始终从当前要下棋的玩家视角计算")
        print()
        print("建议的修复方向：")
        print("   在 _minimax 函数中，不要用 player 来判断胜负，")
        print("   而是用当前轮到谁下棋来判断")

    print("\n" + "=" * 80)
