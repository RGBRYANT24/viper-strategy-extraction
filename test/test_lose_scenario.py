"""
测试后手输掉时的奖励返回
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gym_env
import gymnasium as gym
import numpy as np


def test_player_as_o_loses():
    """测试玩家作为后手输掉的情况"""
    print("=" * 70)
    print("测试：玩家作为后手(O)输掉时的奖励")
    print("=" * 70)

    # 创建环境，对手是 minmax（强对手）
    env = gym.make('TicTacToe-v0', opponent_type='minmax', play_as_o_prob=1.0)

    # 测试多局
    losses_with_neg_reward = 0
    total_losses = 0

    for episode in range(50):
        obs, _ = env.reset()
        done = False

        # 确认玩家是后手
        assert env.unwrapped.play_as_o == True, "玩家应该是后手"

        episode_reward = 0
        step_count = 0

        while not done:
            # 随机选择合法动作（故意不使用最优策略，增加输的概率）
            legal_actions = np.where(obs == 0)[0]
            if len(legal_actions) == 0:
                break
            action = np.random.choice(legal_actions)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward = reward
            step_count += 1

        # 统计输掉的情况
        if 'winner' in info and info['winner'] == 'opponent':
            total_losses += 1
            if episode_reward == -1:
                losses_with_neg_reward += 1
            else:
                print(f"⚠️ 警告：第{episode+1}局输了但奖励不是-1: {episode_reward}")
                print(f"   info: {info}")

    env.close()

    print(f"\n结果:")
    print(f"  总局数: 50")
    print(f"  输掉局数: {total_losses}")
    print(f"  输掉且返回-1的局数: {losses_with_neg_reward}")

    if total_losses > 0:
        if losses_with_neg_reward == total_losses:
            print(f"\n✅ 通过！所有输掉的局都正确返回了 -1 奖励")
        else:
            print(f"\n❌ 失败！有 {total_losses - losses_with_neg_reward} 局输了但没返回 -1")
    else:
        print(f"\n⚠️ 警告：没有输掉的局（这在对战 minmax 时很不正常）")

    print("=" * 70)


def test_player_as_x_loses():
    """测试玩家作为先手(X)输掉的情况"""
    print("\n" + "=" * 70)
    print("测试：玩家作为先手(X)输掉时的奖励")
    print("=" * 70)

    # 创建环境，对手是 minmax（强对手）
    env = gym.make('TicTacToe-v0', opponent_type='minmax', play_as_o_prob=0.0)

    # 测试多局
    losses_with_neg_reward = 0
    total_losses = 0

    for episode in range(50):
        obs, _ = env.reset()
        done = False

        # 确认玩家是先手
        assert env.unwrapped.play_as_o == False, "玩家应该是先手"

        episode_reward = 0

        while not done:
            # 随机选择合法动作
            legal_actions = np.where(obs == 0)[0]
            if len(legal_actions) == 0:
                break
            action = np.random.choice(legal_actions)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward = reward

        # 统计输掉的情况
        if 'winner' in info and info['winner'] == 'opponent':
            total_losses += 1
            if episode_reward == -1:
                losses_with_neg_reward += 1
            else:
                print(f"⚠️ 警告：第{episode+1}局输了但奖励不是-1: {episode_reward}")
                print(f"   info: {info}")

    env.close()

    print(f"\n结果:")
    print(f"  总局数: 50")
    print(f"  输掉局数: {total_losses}")
    print(f"  输掉且返回-1的局数: {losses_with_neg_reward}")

    if total_losses > 0:
        if losses_with_neg_reward == total_losses:
            print(f"\n✅ 通过！所有输掉的局都正确返回了 -1 奖励")
        else:
            print(f"\n❌ 失败！有 {total_losses - losses_with_neg_reward} 局输了但没返回 -1")
    else:
        print(f"\n⚠️ 警告：没有输掉的局")

    print("=" * 70)


if __name__ == "__main__":
    test_player_as_o_loses()
    test_player_as_x_loses()

    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)
