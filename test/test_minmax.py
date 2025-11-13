#!/usr/bin/env python3
"""测试MinMax算法是否正常工作"""

import numpy as np
import gym_env
import gymnasium as gym


def test_minmax_vs_random():
    """测试minmax对战random"""
    env = gym.make('TicTacToe-v0', opponent_type='minmax')

    wins = 0
    losses = 0
    draws = 0

    n_episodes = 100

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            # 随机选择合法动作
            legal_actions = np.where(obs == 0)[0]
            if len(legal_actions) == 0:
                break

            before_action_board = obs.copy()

            action = np.random.choice(legal_actions)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if done:
                if reward > 0:
                    wins += 1
                elif reward < 0:
                    losses += 1
                else:
                    draws += 1

            else:
                print('before action:', before_action_board)
                print('after minmax action:', obs)


    print("Random vs MinMax:")
    print(f"  Wins: {wins} ({wins/n_episodes:.1%})")
    print(f"  Draws: {draws} ({draws/n_episodes:.1%})")
    print(f"  Losses: {losses} ({losses/n_episodes:.1%})")
    print()

    # MinMax should never lose (should win or draw)
    if losses > wins:
        print("✗ 问题: MinMax输的比赢的多，可能有bug")
    elif wins == 0:
        print("✓ MinMax从不输给随机策略，看起来正常")
    else:
        print(f"✓ MinMax赢了{wins}局，输了{losses}局")


def test_minmax_perspective():
    """测试MinMax是否从正确的视角计算"""
    env = gym.make('TicTacToe-v0', opponent_type='minmax')

    # 设置一个即将获胜的局面
    # X (1) 已经有两个连线，只差一步
    # . X .
    # X O .
    # . . .
    obs = np.array([0, 1, 0, 1, -1, 0, 0, 0, 0], dtype=np.float32)
    env.board = obs.copy()
    env.done = False

    print("测试局面（X即将获胜）:")
    env.render()

    # MinMax(-1/O)应该阻止X获胜
    opponent_action = env._minmax_move(obs.copy(), -1)
    print(f"MinMax选择动作: {opponent_action}")

    # 期望：MinMax应该选择位置6（阻止X的0-3-6连线）
    if opponent_action == 6:
        print("✓ MinMax正确阻止了X的获胜")
    else:
        print(f"✗ MinMax选择了{opponent_action}，但应该选择6来阻止X")
    print()

    # 测试另一个场景：MinMax自己即将获胜
    # X . X
    # . O .
    # . O .
    obs2 = np.array([1, 0, 1, 0, -1, 0, 0, -1, 0], dtype=np.float32)
    env.board = obs2.copy()

    print("测试局面（O即将获胜）:")
    env.render()

    opponent_action2 = env._minmax_move(obs2.copy(), -1)
    print(f"MinMax选择动作: {opponent_action2}")

    # 期望：MinMax应该选择位置1（完成4-7-1的连线）或其他获胜位置
    winning_moves = [1]  # 可能的获胜位置
    if opponent_action2 in winning_moves:
        print("✓ MinMax正确选择了获胜位置")
    else:
        print(f"✗ MinMax选择了{opponent_action2}，但应该选择获胜位置")


if __name__ == "__main__":
    print("="*80)
    print("MinMax算法测试")
    print("="*80)
    print()

    test_minmax_vs_random()
    print()
    test_minmax_perspective()

    print("\n" + "="*80)
    print("测试完成")
    print("="*80)
