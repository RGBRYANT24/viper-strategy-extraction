"""
使用自我对弈训练TicTacToe
"""
import argparse
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import gym_env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--n-env", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--update-interval", type=int, default=10000,
                        help="每多少步更新一次对手策略")
    parser.add_argument("--output", type=str, default="log/oracle_TicTacToe_selfplay.zip")
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    print("=" * 70)
    print("TicTacToe 自我对弈训练")
    print("=" * 70)
    print(f"总步数: {args.total_timesteps}")
    print(f"并行环境数: {args.n_env}")
    print(f"对手更新间隔: {args.update_interval}")
    print()

    # 创建环境
    def make_selfplay_env():
        env = gym.make('TicTacToe-SelfPlay-v0')
        return Monitor(env)

    envs = DummyVecEnv([make_selfplay_env for _ in range(args.n_env)])

    # 创建模型
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
        exploration_fraction=0.5,
        exploration_final_eps=0.1,
        exploration_initial_eps=1.0,
        policy_kwargs={'net_arch': [128, 128]},
        verbose=args.verbose,
        seed=args.seed
    )

    print("开始自我对弈训练...")
    print("阶段1：对战随机对手（热身）")

    # 分阶段训练
    steps_trained = 0
    update_count = 0

    while steps_trained < args.total_timesteps:
        steps_this_round = min(args.update_interval, args.total_timesteps - steps_trained)

        # 训练
        model.learn(
            total_timesteps=steps_this_round,
            reset_num_timesteps=False,
            log_interval=100
        )

        steps_trained += steps_this_round
        update_count += 1

        # 更新对手策略（用当前策略的副本）
        print(f"\n[更新 {update_count}] 已训练 {steps_trained}/{args.total_timesteps} 步")
        print("更新对手策略为当前策略...")

        # 更新每个环境的对手
        for env_idx in range(args.n_env):
            try:
                # 获取单个环境
                single_env = envs.envs[env_idx].env
                # 设置对手为当前模型
                single_env.set_opponent_policy(model)
                print(f"  环境 {env_idx + 1}/{args.n_env} 对手已更新")
            except Exception as e:
                print(f"  环境 {env_idx + 1} 更新失败: {e}")

        print()

    # 保存模型
    print(f"\n训练完成！保存模型到 {args.output}")
    model.save(args.output)

    # 测试
    print("\n" + "=" * 70)
    print("测试模型（对战MinMax）")
    print("=" * 70)

    test_env = gym.make('TicTacToe-v0', opponent_type='minmax')
    test_env = Monitor(test_env)

    wins = 0
    losses = 0
    draws = 0

    for i in range(50):
        obs, _ = test_env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated

            if done:
                if reward > 0:
                    wins += 1
                elif reward < 0:
                    losses += 1
                else:
                    draws += 1

    print(f"\n结果 (50局):")
    print(f"  胜: {wins} ({wins/50*100:.1f}%)")
    print(f"  负: {losses} ({losses/50*100:.1f}%)")
    print(f"  平: {draws} ({draws/50*100:.1f}%)")

    if draws >= 40:
        print("\n✓ 优秀！高平局率说明学到了接近最优策略。")
    elif draws >= 30:
        print("\n△ 良好，但还有提升空间。")
    else:
        print("\n⚠ 需要更多训练或调整参数。")

if __name__ == "__main__":
    main()
