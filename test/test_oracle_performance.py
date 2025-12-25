"""
测试 Oracle PPO 模型的性能
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gym_env  # 注册环境
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import numpy as np
import argparse


def mask_fn(env):
    """获取动作掩码"""
    # 获取最底层的原始环境
    unwrapped_env = env.unwrapped
    return (unwrapped_env.board == 0).astype(np.int8)


def test_oracle(oracle_path, opponent_type='minmax', play_as_o_prob=0.5, n_episodes=100):
    """
    测试 Oracle 模型性能

    Args:
        oracle_path: Oracle 模型路径
        opponent_type: 对手类型 ('random' or 'minmax')
        play_as_o_prob: 作为后手的概率 (0.0=总是先手, 0.5=随机, 1.0=总是后手)
        n_episodes: 测试局数
    """
    print("=" * 70)
    print(f"测试 Oracle 性能")
    print("=" * 70)
    print(f"Oracle 路径: {oracle_path}")
    print(f"对手类型: {opponent_type}")
    print(f"先后手设置: play_as_o_prob={play_as_o_prob}")
    if play_as_o_prob == 0.0:
        print("  (总是先手)")
    elif play_as_o_prob == 1.0:
        print("  (总是后手)")
    elif play_as_o_prob == 0.5:
        print("  (随机先后手)")
    print(f"测试局数: {n_episodes}")
    print()

    # 创建环境
    env = gym.make('TicTacToe-v0',
                   opponent_type=opponent_type,
                   play_as_o_prob=play_as_o_prob)
    env = ActionMasker(env, mask_fn)

    # 加载 Oracle
    print("加载 Oracle 模型...")
    oracle = MaskablePPO.load(oracle_path, env=env)
    print("✓ Oracle 加载成功")
    print()

    # 测试
    print(f"开始测试 vs {opponent_type.upper()}...")
    print()

    wins, draws, losses = 0, 0, 0
    illegal_moves = 0

    # 统计先后手的胜负
    stats_as_x = {'wins': 0, 'draws': 0, 'losses': 0, 'count': 0}
    stats_as_o = {'wins': 0, 'draws': 0, 'losses': 0, 'count': 0}

    for i in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        # 记录本局是先手还是后手
        is_o = env.unwrapped.play_as_o

        while not done:
            mask = mask_fn(env)
            action, _ = oracle.predict(obs, deterministic=True, action_masks=mask)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward = reward

            # 检查非法移动
            if 'illegal_move' in info and info['illegal_move']:
                illegal_moves += 1

        # 统计结果
        if episode_reward > 0:
            wins += 1
            if is_o:
                stats_as_o['wins'] += 1
            else:
                stats_as_x['wins'] += 1
        elif episode_reward < 0:
            losses += 1
            if is_o:
                stats_as_o['losses'] += 1
            else:
                stats_as_x['losses'] += 1
        else:
            draws += 1
            if is_o:
                stats_as_o['draws'] += 1
            else:
                stats_as_x['draws'] += 1

        if is_o:
            stats_as_o['count'] += 1
        else:
            stats_as_x['count'] += 1

        # 进度显示
        if (i+1) % 20 == 0:
            print(f"  进度: {i+1}/{n_episodes} - 胜{wins} 平{draws} 负{losses}")

    env.close()

    # 打印结果
    print()
    print("=" * 70)
    print(f"测试结果 ({n_episodes}局 vs {opponent_type.upper()})")
    print("=" * 70)
    print(f"总体结果:")
    print(f"  胜: {wins} ({wins/n_episodes*100:.1f}%)")
    print(f"  平: {draws} ({draws/n_episodes*100:.1f}%)")
    print(f"  负: {losses} ({losses/n_episodes*100:.1f}%)")

    if illegal_moves > 0:
        print(f"  ⚠ 非法移动: {illegal_moves}")

    print()

    # 分别显示先后手的结果
    if stats_as_x['count'] > 0:
        print(f"作为先手(X) - {stats_as_x['count']}局:")
        print(f"  胜: {stats_as_x['wins']} ({stats_as_x['wins']/stats_as_x['count']*100:.1f}%)")
        print(f"  平: {stats_as_x['draws']} ({stats_as_x['draws']/stats_as_x['count']*100:.1f}%)")
        print(f"  负: {stats_as_x['losses']} ({stats_as_x['losses']/stats_as_x['count']*100:.1f}%)")

    if stats_as_o['count'] > 0:
        print(f"\n作为后手(O) - {stats_as_o['count']}局:")
        print(f"  胜: {stats_as_o['wins']} ({stats_as_o['wins']/stats_as_o['count']*100:.1f}%)")
        print(f"  平: {stats_as_o['draws']} ({stats_as_o['draws']/stats_as_o['count']*100:.1f}%)")
        print(f"  负: {stats_as_o['losses']} ({stats_as_o['losses']/stats_as_o['count']*100:.1f}%)")

    print()

    # 评估
    if illegal_moves > 0:
        print("⚠ 有非法移动 - 模型可能有问题")
    elif opponent_type == 'minmax':
        if draws >= n_episodes * 0.8:
            print("✓ 优秀！高平局率说明学到了接近最优策略")
        elif draws >= n_episodes * 0.6:
            print("△ 良好，但还有提升空间")
        else:
            print("⚠ 需要更多训练")
    elif opponent_type == 'random':
        if wins >= n_episodes * 0.9:
            print("✓ 优秀！能够稳定战胜随机对手")
        elif wins >= n_episodes * 0.7:
            print("△ 良好，但还有提升空间")
        else:
            print("⚠ 需要更多训练")

    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='测试 Oracle 模型性能')
    parser.add_argument('--oracle-path', type=str,
                       default='log/oracle_TicTacToe_ppo_aggressive.zip',
                       help='Oracle 模型路径')
    parser.add_argument('--opponent', type=str, default='minmax',
                       choices=['random', 'minmax'],
                       help='对手类型')
    parser.add_argument('--play-as-o-prob', type=float, default=0.5,
                       help='作为后手的概率 (0.0=先手, 0.5=随机, 1.0=后手)')
    parser.add_argument('--n-episodes', type=int, default=100,
                       help='测试局数')

    args = parser.parse_args()

    test_oracle(
        oracle_path=args.oracle_path,
        opponent_type=args.opponent,
        play_as_o_prob=args.play_as_o_prob,
        n_episodes=args.n_episodes
    )
