"""
测试训练好的模型
"""

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gym_env


def mask_fn(env):
    """返回 action mask"""
    # 获取棋盘状态 - 尝试多种方式解包环境
    board = None

    # 方法 1: 直接访问 board
    if hasattr(env, 'board'):
        board = env.board
    # 方法 2: 通过 unwrapped 访问
    elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'board'):
        board = env.unwrapped.board
    # 方法 3: 通过 env.env 访问
    elif hasattr(env, 'env'):
        if hasattr(env.env, 'board'):
            board = env.env.board
        elif hasattr(env.env, 'unwrapped') and hasattr(env.env.unwrapped, 'board'):
            board = env.env.unwrapped.board
        elif hasattr(env.env, 'env') and hasattr(env.env.env, 'board'):
            board = env.env.env.board

    if board is None:
        raise AttributeError(f"Cannot find 'board' attribute in environment. Env type: {type(env)}")

    mask = (board == 0).astype(np.int8)
    return mask


def test_model(model_path, opponent_type='minmax', num_games=100, render=False):
    """
    测试模型性能

    Args:
        model_path: 模型文件路径
        opponent_type: 对手类型 ('random' 或 'minmax')
        num_games: 测试局数
        render: 是否渲染游戏过程
    """
    print("=" * 70)
    print(f"测试模型: {model_path}")
    print(f"对手类型: {opponent_type}")
    print(f"测试局数: {num_games}")
    print("=" * 70)

    # 创建环境
    env = gym.make('TicTacToe-v0', opponent_type=opponent_type)
    env = Monitor(env)
    env = ActionMasker(env, mask_fn)

    # 加载模型
    model = MaskablePPO.load(model_path, env=env)
    print(f"✓ 模型加载成功")
    print()

    # 测试
    wins, losses, draws, illegal = 0, 0, 0, 0

    for i in range(num_games):
        obs, _ = env.reset()
        done = False
        game_steps = 0

        if render and i < 3:  # 只渲染前3局
            print(f"\n=== 第 {i+1} 局 ===")

        while not done:
            # 获取 mask 并预测
            action_mask = mask_fn(env)
            action, _ = model.predict(obs, deterministic=True, action_masks=action_mask)

            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            game_steps += 1

            if render and i < 3:
                env.render()

            if done:
                if 'illegal_move' in info and info['illegal_move']:
                    illegal += 1
                    losses += 1
                    if render and i < 3:
                        print(f"❌ 非法移动导致失败")
                elif reward > 0:
                    wins += 1
                    if render and i < 3:
                        print(f"✓ 获胜！")
                elif reward < 0:
                    losses += 1
                    if render and i < 3:
                        print(f"✗ 失败")
                else:
                    draws += 1
                    if render and i < 3:
                        print(f"= 平局")

        # 每 10 局显示一次进度
        if (i + 1) % 10 == 0:
            print(f"已完成 {i+1}/{num_games} 局...")

    env.close()

    # 统计结果
    print("\n" + "=" * 70)
    print(f"测试结果 ({num_games} 局 vs {opponent_type.upper()})")
    print("=" * 70)
    print(f"胜: {wins:3d} ({wins/num_games*100:5.1f}%)")
    print(f"负: {losses:3d} ({losses/num_games*100:5.1f}%)")
    print(f"平: {draws:3d} ({draws/num_games*100:5.1f}%)")
    print(f"非法移动: {illegal:3d}")
    print("=" * 70)

    # 评估
    if illegal > 0:
        print("⚠️  有非法移动 - masking 未正常工作")
    elif opponent_type == 'minmax':
        if draws >= num_games * 0.8:
            print("✓ 优秀！高平局率说明学到了接近最优策略")
        elif draws >= num_games * 0.6:
            print("△ 良好，但还有提升空间")
        else:
            print("⚠️  需要更多训练或调整参数")
    else:  # random opponent
        if wins >= num_games * 0.8:
            print("✓ 优秀！对战随机对手胜率很高")
        elif wins >= num_games * 0.6:
            print("△ 良好，但还有提升空间")
        else:
            print("⚠️  对战随机对手应该有更高胜率")

    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='测试 TicTacToe 模型')
    parser.add_argument('--model', type=str, default='log/oracle_TicTacToe_ppo_masked.zip',
                        help='模型文件路径')
    parser.add_argument('--opponent', type=str, default='minmax',
                        choices=['random', 'minmax'], help='对手类型')
    parser.add_argument('--num-games', type=int, default=100,
                        help='测试局数')
    parser.add_argument('--render', action='store_true',
                        help='是否渲染前几局游戏')
    args = parser.parse_args()

    test_model(args.model, args.opponent, args.num_games, args.render)
