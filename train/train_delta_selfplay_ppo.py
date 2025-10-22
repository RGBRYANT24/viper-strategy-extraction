"""
使用 MaskablePPO 训练 TicTacToe Delta-Uniform Self-Play
支持 action masking，解决 Q值污染问题
"""

import argparse
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import copy
from collections import deque
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gym_env
from gym_env.tictactoe_delta_selfplay import TicTacToeDeltaSelfPlayEnv
from gym_env.policies import RandomPlayerPolicy, MinMaxPlayerPolicy
from gymnasium import spaces


def mask_fn(env):
    """
    返回 action mask

    Args:
        env: 环境实例

    Returns:
        mask: numpy array of shape (9,), 1=legal, 0=illegal
    """
    # 获取棋盘状态
    if hasattr(env, 'board'):
        board = env.board
    else:
        # 如果是包装过的环境
        board = env.env.board

    # 合法动作 mask
    mask = (board == 0).astype(np.int8)
    return mask


def main():
    parser = argparse.ArgumentParser(description='TicTacToe MaskablePPO Training')
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--n-env", type=int, default=8)
    parser.add_argument("--update-interval", type=int, default=10000)
    parser.add_argument("--max-pool-size", type=int, default=20)
    parser.add_argument("--play-as-o-prob", type=float, default=0.5)
    parser.add_argument("--output", type=str, default="log/oracle_TicTacToe_ppo.zip")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--use-minmax", action="store_true")
    parser.add_argument("--net-arch", type=str, default="128,128")
    args = parser.parse_args()

    print("=" * 70)
    print(f"TicTacToe MaskablePPO Training (Delta-{args.max_pool_size}-Uniform)")
    print("=" * 70)
    print(f"总步数: {args.total_timesteps}")
    print(f"并行环境数: {args.n_env}")
    print(f"✓ 使用 Action Masking")
    print()

    # --- 步骤 1: 初始化策略池 ---
    temp_env = TicTacToeDeltaSelfPlayEnv()
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    baseline_policies = [RandomPlayerPolicy(obs_space, act_space)]
    if args.use_minmax:
        baseline_policies.append(MinMaxPlayerPolicy(obs_space, act_space))

    learned_policy_pool = deque(maxlen=args.max_pool_size)

    print(f"基准策略池: {len(baseline_policies)} 个对手")
    print(f"学习策略池: 容量 = {args.max_pool_size}")
    print()

    # --- 步骤 2: 创建环境（带 Action Masking）---
    def make_masked_env():
        env = TicTacToeDeltaSelfPlayEnv(
            baseline_pool=baseline_policies,
            learned_pool=learned_policy_pool,
            play_as_o_prob=args.play_as_o_prob,
            sampling_strategy='uniform'
        )
        env = Monitor(env)
        # ⭐ 关键：添加 ActionMasker 包装器
        env = ActionMasker(env, mask_fn)
        return env

    envs = DummyVecEnv([make_masked_env for _ in range(args.n_env)])

    print(f"创建了 {args.n_env} 个并行环境（带 masking）")
    print()

    # --- 步骤 3: 创建 MaskablePPO 模型 ---
    net_arch = [int(x) for x in args.net_arch.split(',')]

    model = MaskablePPO(
        policy='MlpPolicy',
        env=envs,
        learning_rate=1e-3,
        n_steps=128,           # PPO: 每次收集多少步
        batch_size=64,
        n_epochs=10,           # PPO: 每批数据训练几轮
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,         # 熵系数，鼓励探索
        policy_kwargs={'net_arch': net_arch},
        verbose=args.verbose,
        seed=args.seed
    )

    print("MaskablePPO 模型配置:")
    print(f"  网络结构: {net_arch}")
    print(f"  学习率: {model.learning_rate}")
    print(f"  批大小: {model.batch_size}")
    print(f"  ✓ Action Masking: 启用")
    print()

    # --- 步骤 4: 训练循环 ---
    print("=" * 70)
    print("开始训练（MaskablePPO + Delta-Uniform Self-Play）")
    print("=" * 70)
    print()

    steps_trained = 0
    update_count = 0

    while steps_trained < args.total_timesteps:
        steps_this_round = min(args.update_interval, args.total_timesteps - steps_trained)

        print(f"[训练轮次 {update_count + 1}] 训练 {steps_this_round} 步...")
        model.learn(
            total_timesteps=steps_this_round,
            reset_num_timesteps=False,
            log_interval=100
        )

        steps_trained += steps_this_round
        update_count += 1

        print(f"\n[更新 {update_count}] 已训练 {steps_trained}/{args.total_timesteps} 步")
        print("更新对手池...")

        current_policy_snapshot = copy.deepcopy(model.policy)
        learned_policy_pool.append(current_policy_snapshot)

        print(f"  学习策略池: {len(learned_policy_pool)}/{args.max_pool_size}")
        print()

    # --- 步骤 5: 保存模型 ---
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    model.save(args.output)

    print()
    print("=" * 70)
    print("训练完成！")
    print("=" * 70)
    print(f"模型已保存到: {args.output}")
    print()

    # --- 步骤 6: 测试 ---
    print("=" * 70)
    print("测试模型 - 对战 MinMax")
    print("=" * 70)

    test_env = gym.make('TicTacToe-v0', opponent_type='minmax')
    test_env = Monitor(test_env)
    test_env = ActionMasker(test_env, mask_fn)

    wins, losses, draws, illegal = 0, 0, 0, 0

    for _ in range(50):
        obs, _ = test_env.reset()
        done = False

        while not done:
            # MaskablePPO 自动处理 masking
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated

            if done:
                if 'illegal_move' in info and info['illegal_move']:
                    illegal += 1
                    losses += 1
                elif reward > 0:
                    wins += 1
                elif reward < 0:
                    losses += 1
                else:
                    draws += 1

    test_env.close()

    print(f"\n结果 (50局 vs MinMax):")
    print(f"  胜: {wins} ({wins/50*100:.1f}%)")
    print(f"  负: {losses} ({losses/50*100:.1f}%)")
    print(f"  平: {draws} ({draws/50*100:.1f}%)")
    print(f"  非法移动: {illegal}")
    print()

    if illegal > 0:
        print("⚠ 有非法移动 - 检查 masking 是否正常工作")
    elif draws >= 40:
        print("✓ 优秀！高平局率说明学到了接近最优策略。")
    elif draws >= 30:
        print("△ 良好，但还有提升空间。")
    else:
        print("⚠ 需要更多训练或调整参数。")


if __name__ == "__main__":
    main()
