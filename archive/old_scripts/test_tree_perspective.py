"""
测试决策树的视角转换是否正确
"""

import numpy as np
import joblib
import sys

def test_perspective():
    """测试视角转换"""

    print("="*70)
    print("测试决策树视角转换")
    print("="*70)
    print()

    # 加载决策树
    tree_path = "log/viper_TicTacToe-v0_100_15.joblib"

    try:
        model = joblib.load(tree_path)
        print(f"✓ 加载模型: {tree_path}")
        print(f"  类型: {type(model)}")

        if hasattr(model, 'tree'):
            tree = model.tree
            print(f"  这是TreeWrapper，内部树: {type(tree)}")
            model = tree

        print(f"  叶子节点数: {model.get_n_leaves()}")
        print(f"  树深度: {model.get_depth()}")
        print()
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return False

    # 测试场景1: X玩家视角（正常训练视角）
    print("测试1: X玩家视角 (训练时的视角)")
    print("-"*70)

    # 棋盘状态：X在中间，O在角落
    board_x_view = np.array([
        -1,  0,  0,   # O . .
         0,  1,  0,   # . X .
         0,  0,  0    # . . .
    ], dtype=np.float32)

    print("棋盘 (X视角):")
    print("  O . .")
    print("  . X .")
    print("  . . .")
    print(f"  数组: {board_x_view}")
    print()

    action_x = model.predict(board_x_view.reshape(1, -1))[0]
    legal_actions_x = np.where(board_x_view == 0)[0]

    print(f"  预测动作: {action_x}")
    print(f"  合法动作: {legal_actions_x}")
    print(f"  是否合法: {action_x in legal_actions_x}")
    print()

    # 测试场景2: O玩家视角（需要翻转）
    print("测试2: O玩家视角 (对战时作为O)")
    print("-"*70)

    # 从环境视角看：X在角落，O在中间
    board_env_view = np.array([
         1,  0,  0,   # X . .
         0, -1,  0,   # . O .
         0,  0,  0    # . . .
    ], dtype=np.float32)

    print("棋盘 (环境视角，O是当前玩家):")
    print("  X . .")
    print("  . O .")
    print("  . . .")
    print(f"  数组: {board_env_view}")
    print()

    # 方法1: 不翻转（错误）
    print("方法1: 不翻转视角 (错误方法)")
    action_wrong = model.predict(board_env_view.reshape(1, -1))[0]
    legal_actions = np.where(board_env_view == 0)[0]
    print(f"  预测动作: {action_wrong}")
    print(f"  合法动作: {legal_actions}")
    print(f"  是否合法: {action_wrong in legal_actions}")
    print()

    # 方法2: 翻转视角（正确）
    print("方法2: 翻转视角 (正确方法)")
    board_o_view = -board_env_view  # O变1，X变-1
    print(f"  转换后棋盘: {board_o_view}")
    print("  转换后 (O视角):")
    board_2d = board_o_view.reshape(3, 3)
    for row in board_2d:
        print("  " + " ".join(["O" if x == 1 else "X" if x == -1 else "." for x in row]))

    action_correct = model.predict(board_o_view.reshape(1, -1))[0]
    print(f"  预测动作: {action_correct}")
    print(f"  合法动作: {legal_actions}")
    print(f"  是否合法: {action_correct in legal_actions}")
    print()

    # 总结
    print("="*70)
    print("结论:")
    print("="*70)

    if action_wrong in legal_actions:
        print("⚠️  不翻转也能工作？这很奇怪，可能模型有问题")
    else:
        print("✓ 不翻转视角会导致非法动作（预期行为）")

    if action_correct in legal_actions:
        print("✓ 翻转视角后可以得到合法动作（正确！）")
    else:
        print("❌ 即使翻转视角也无法得到合法动作（模型质量问题）")

    print()
    print("修复建议:")
    print("  1. 确保 battle_nn_vs_tree.py 中 DecisionTreePlayer.predict() 有 player_id 参数")
    print("  2. 确保调用时传入正确的 player_id (-1 for O, 1 for X)")
    print("  3. 清除Python缓存: rm -rf __pycache__")
    print("  4. 重新运行对战")

    return action_correct in legal_actions


if __name__ == "__main__":
    success = test_perspective()
    sys.exit(0 if success else 1)
