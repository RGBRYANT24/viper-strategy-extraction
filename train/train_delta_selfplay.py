"""
使用 Delta-Uniform Self-Play 训练 TicTacToe
解决普通自我对弈陷入局部最优的问题

核心思想:
1. 维护固定大小的历史策略池 (K个历史快照)
2. 加入基准策略池 (MinMax, Random) 提高鲁棒性
3. 每次 reset() 时从两个池中均匀采样对手
4. 训练先手和后手，通过棋盘翻转保持网络输入一致性
"""

import argparse
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import copy
from collections import deque
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gym_env
from gym_env.tictactoe_delta_selfplay import TicTacToeDeltaSelfPlayEnv
from gym_env.policies import RandomPlayerPolicy, MinMaxPlayerPolicy


def main():
    parser = argparse.ArgumentParser(description='TicTacToe Delta-Uniform Self-Play Training')
    parser.add_argument("--total-timesteps", type=int, default=200000,
                        help="总训练步数")
    parser.add_argument("--n-env", type=int, default=8,
                        help="并行环境数")
    parser.add_argument("--update-interval", type=int, default=10000,
                        help="每多少步更新一次策略池")
    parser.add_argument("--max-pool-size", type=int, default=20,
                        help="历史策略池的最大容量 (K)")
    parser.add_argument("--play-as-o-prob", type=float, default=0.5,
                        help="作为后手(O)的概率，默认0.5 (先后手各50%)")
    parser.add_argument("--output", type=str, default="log/oracle_TicTacToe_delta_selfplay.zip",
                        help="模型保存路径")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--verbose", type=int, default=1,
                        help="详细输出级别")
    parser.add_argument("--use-minmax", action="store_true",
                        help="在基准池中包含 MinMax 策略")
    parser.add_argument("--net-arch", type=str, default="128,128",
                        help="网络结构，用逗号分隔，如 '128,128' 或 '256,256'")
    args = parser.parse_args()

    print("=" * 70)
    print(f"TicTacToe Delta-{args.max_pool_size}-Uniform Self-Play 训练")
    print("=" * 70)
    print(f"总步数: {args.total_timesteps}")
    print(f"并行环境数: {args.n_env}")
    print(f"策略池更新间隔: {args.update_interval}")
    print(f"历史策略池容量 (K): {args.max_pool_size}")
    print(f"后手训练概率: {args.play_as_o_prob}")
    print(f"使用 MinMax 基准: {args.use_minmax}")
    print()

    # --- 步骤 1: 初始化策略池 ---

    # 创建临时环境获取空间信息
    temp_env = TicTacToeDeltaSelfPlayEnv()
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    # 创建基准策略池
    baseline_policies = [
        RandomPlayerPolicy(obs_space, act_space)
    ]

    if args.use_minmax:
        baseline_policies.append(MinMaxPlayerPolicy(obs_space, act_space))

    print(f"初始化基准策略池: {len(baseline_policies)} 个基准对手")
    for i, policy in enumerate(baseline_policies):
        print(f"  [{i+1}] {policy.__class__.__name__}")

    # 创建学习策略池（固定大小的双端队列）
    learned_policy_pool = deque(maxlen=args.max_pool_size)
    print(f"初始化学习策略池: 容量 = {args.max_pool_size}")
    print()

    # --- 步骤 2: 创建环境 ---

    def make_delta_selfplay_env():
        """创建 Delta-Uniform Self-Play 环境"""
        env = TicTacToeDeltaSelfPlayEnv(
            baseline_pool=baseline_policies,
            learned_pool=learned_policy_pool,
            play_as_o_prob=args.play_as_o_prob,
            sampling_strategy='uniform'
        )
        return Monitor(env)

    # 创建向量化环境
    envs = DummyVecEnv([make_delta_selfplay_env for _ in range(args.n_env)])

    print(f"创建了 {args.n_env} 个并行环境")
    print()

    # --- 步骤 3: 创建 DQN 模型 ---

    # 解析网络结构
    net_arch = [int(x) for x in args.net_arch.split(',')]

    model = DQN(
        policy='MlpPolicy',
        env=envs,
        learning_starts=1000,
        learning_rate=1e-3,
        buffer_size=100_000,
        batch_size=64,
        target_update_interval=500,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.5,  # 前50%时间进行探索
        exploration_final_eps=0.05,  # 最终探索率5%
        exploration_initial_eps=1.0,
        policy_kwargs={'net_arch': net_arch},
        verbose=args.verbose,
        seed=args.seed
    )

    print("DQN 模型配置:")
    print(f"  网络结构: {model.policy_kwargs['net_arch']}")
    print(f"  学习率: {model.learning_rate}")
    print(f"  批大小: {model.batch_size}")
    print(f"  经验回放池: {model.buffer_size}")
    print(f"  探索策略: ε-greedy ({model.exploration_initial_eps} → {model.exploration_final_eps})")
    print()

    # --- 步骤 4: Delta-Uniform Self-Play 训练循环 ---

    print("=" * 70)
    print("开始 Delta-Uniform Self-Play 训练...")
    print("=" * 70)
    print()

    steps_trained = 0
    update_count = 0

    while steps_trained < args.total_timesteps:
        steps_this_round = min(args.update_interval, args.total_timesteps - steps_trained)

        # 训练 (环境内部会在 reset() 时自动从池中采样对手)
        print(f"[训练轮次 {update_count + 1}] 训练 {steps_this_round} 步...")
        model.learn(
            total_timesteps=steps_this_round,
            reset_num_timesteps=False,
            log_interval=100
        )

        steps_trained += steps_this_round
        update_count += 1

        # --- 步骤 5: 更新学习策略池 ---
        print()
        print(f"[更新 {update_count}] 已训练 {steps_trained}/{args.total_timesteps} 步")
        print("将当前策略快照添加到对手池...")

        # 深拷贝当前策略
        current_policy_snapshot = copy.deepcopy(model.policy)

        # 添加到固定大小的队列
        # 如果队列已满，最旧的策略会被自动淘汰
        learned_policy_pool.append(current_policy_snapshot)

        print(f"  学习策略池大小: {len(learned_policy_pool)} / {args.max_pool_size}")
        print(f"  总对手数量: {len(baseline_policies)} (基准) + {len(learned_policy_pool)} (学习) = {len(baseline_policies) + len(learned_policy_pool)}")
        print()

    # --- 步骤 6: 保存模型 ---
    print()
    print("=" * 70)
    print("训练完成！")
    print("=" * 70)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    model.save(args.output)
    print(f"模型已保存到: {args.output}")
    print()

    # --- 步骤 7: 测试评估 ---
    print("=" * 70)
    print("测试模型 - 对战 MinMax")
    print("=" * 70)
    print()

    test_env = gym.make('TicTacToe-v0', opponent_type='minmax')
    test_env = Monitor(test_env)

    wins = 0
    losses = 0
    draws = 0
    illegal_moves = 0

    num_test_games = 50
    print(f"进行 {num_test_games} 局测试...")

    for i in range(num_test_games):
        obs, _ = test_env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated

            if done:
                if 'illegal_move' in info and info['illegal_move']:
                    illegal_moves += 1
                    losses += 1
                elif reward > 0:
                    wins += 1
                elif reward < 0:
                    losses += 1
                else:
                    draws += 1

    test_env.close()

    # 打印结果
    print()
    print(f"测试结果 ({num_test_games} 局 vs MinMax):")
    print(f"  胜: {wins} ({wins/num_test_games*100:.1f}%)")
    print(f"  负: {losses} ({losses/num_test_games*100:.1f}%)")
    print(f"  平: {draws} ({draws/num_test_games*100:.1f}%)")
    print(f"  非法移动: {illegal_moves}")
    print()

    # 评估
    if illegal_moves > 0:
        print("⚠ 存在非法移动！需要检查策略。")
    elif draws >= 40:
        print("✓ 优秀！高平局率说明学到了接近最优策略。")
    elif draws >= 30:
        print("△ 良好，但还有提升空间。")
    else:
        print("⚠ 需要更多训练或调整参数。")

    print()
    print("=" * 70)
    print("训练和评估完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
