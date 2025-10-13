"""
调试非法移动：捕获决策树预测非法动作的具体情况
"""

import numpy as np
import joblib
from battle_nn_vs_tree import (
    TicTacToeBattleEnv,
    DecisionTreePlayer,
    NeuralNetPlayer
)

def debug_single_game():
    """调试单局游戏，详细输出每一步"""

    print("="*70)
    print("调试单局游戏")
    print("="*70)
    print()

    # 创建环境和玩家
    env = TicTacToeBattleEnv()
    nn_player = NeuralNetPlayer("log/oracle_TicTacToe_selfplay.zip")
    tree_player = DecisionTreePlayer("log/viper_TicTacToe-v0_100_15.joblib", debug=True)

    print("测试1: 神经网络(X) vs 决策树(O)")
    print("-"*70)

    obs = env.reset()
    illegal_count = 0
    step = 0

    while not env.done and step < 9:
        step += 1

        # 当前玩家
        if env.current_player == 1:
            current_agent = nn_player
            agent_name = "神经网络 (X)"
            player_id = 1
        else:
            current_agent = tree_player
            agent_name = "决策树 (O)"
            player_id = -1

        print(f"\n步骤 {step}: {agent_name} 的回合")
        print(f"当前棋盘:")
        env.render()
        print(f"棋盘数组: {obs}")
        print(f"合法动作: {env.get_legal_actions()}")

        # 预测动作
        if isinstance(current_agent, DecisionTreePlayer):
            print(f"\n决策树预测 (player_id={player_id}):")
            # 显示转换前后的棋盘
            if player_id == -1:
                print(f"  原始棋盘: {obs}")
                obs_transformed = -obs
                print(f"  翻转棋盘: {obs_transformed}")
            else:
                obs_transformed = obs
                print(f"  棋盘(不翻转): {obs}")

            # 直接用模型预测看看
            action_no_flip = tree_player.model.predict(obs.reshape(1, -1))[0]
            action_with_flip = tree_player.model.predict(obs_transformed.reshape(1, -1))[0]

            print(f"  不翻转预测: {action_no_flip}")
            print(f"  翻转后预测: {action_with_flip}")

            # 使用正确的接口
            action = current_agent.predict(obs, player_id=player_id)
        else:
            action = current_agent.predict(obs)

        print(f"\n{agent_name} 选择动作: {action}")

        # 检查是否合法
        legal_actions = env.get_legal_actions()
        is_legal = action in legal_actions

        if not is_legal:
            print(f"❌ 非法动作！")
            print(f"   动作 {action} 不在合法动作 {legal_actions} 中")
            print(f"   位置 {action} 的值: {obs[action]}")
            illegal_count += 1

            # 如果是决策树的非法动作，详细分析
            if isinstance(current_agent, DecisionTreePlayer):
                print(f"\n详细分析:")
                print(f"   决策树看到的棋盘（翻转后）: {obs_transformed}")
                print(f"   决策树预测: {action}")
                print(f"   该位置实际值: {obs[action]}")

                # 尝试找出决策树为什么预测这个动作
                print(f"\n尝试所有合法动作看决策树的偏好:")
                for legal_action in legal_actions:
                    print(f"     动作 {legal_action}: 位置值 = {obs[legal_action]}")

            break
        else:
            print(f"✓ 合法动作")

        # 执行动作
        obs, reward, done, info = env.step(action)

        if done:
            print(f"\n游戏结束!")
            if 'illegal_move' in info:
                print(f"原因: 非法移动")
            elif 'winner' in info:
                print(f"原因: {'X' if info['winner'] == 1 else 'O'} 获胜")
            elif 'draw' in info:
                print(f"原因: 平局")

    print(f"\n总非法移动数: {illegal_count}")
    return illegal_count


def test_multiple_games():
    """测试多局游戏，统计非法移动模式"""

    print("\n" + "="*70)
    print("测试多局游戏")
    print("="*70)
    print()

    env = TicTacToeBattleEnv()
    nn_player = NeuralNetPlayer("log/oracle_TicTacToe_selfplay.zip")
    tree_player = DecisionTreePlayer("log/viper_TicTacToe-v0_100_15.joblib")

    illegal_positions = []  # 记录所有非法动作的位置
    illegal_boards = []     # 记录导致非法动作的棋盘

    n_games = 20
    illegal_count = 0

    for game in range(n_games):
        obs = env.reset()
        step = 0

        while not env.done and step < 9:
            step += 1

            if env.current_player == 1:
                current_agent = nn_player
                player_id = 1
            else:
                current_agent = tree_player
                player_id = -1

            # 预测
            if isinstance(current_agent, DecisionTreePlayer):
                action = current_agent.predict(obs, player_id=player_id)
            else:
                action = current_agent.predict(obs)

            # 检查合法性
            legal_actions = env.get_legal_actions()
            if action not in legal_actions:
                illegal_count += 1
                illegal_positions.append(action)
                illegal_boards.append(obs.copy())
                print(f"游戏 {game+1}, 步骤 {step}: 非法动作 {action}")
                print(f"  棋盘: {obs}")
                print(f"  合法: {legal_actions}")
                break

            obs, reward, done, info = env.step(action)

    print(f"\n统计 ({n_games} 局):")
    print(f"  非法移动次数: {illegal_count}")

    if illegal_count > 0:
        print(f"\n非法动作位置分布:")
        from collections import Counter
        pos_counter = Counter(illegal_positions)
        for pos, count in pos_counter.most_common():
            print(f"    位置 {pos}: {count} 次")

        print(f"\n第一个非法移动的详细信息:")
        print(f"  棋盘: {illegal_boards[0]}")
        print(f"  动作: {illegal_positions[0]}")
        print(f"  该位置的值: {illegal_boards[0][illegal_positions[0]]}")


if __name__ == "__main__":
    # 先调试单局
    illegal = debug_single_game()

    # 如果有非法移动，测试多局找模式
    if illegal > 0:
        test_multiple_games()
