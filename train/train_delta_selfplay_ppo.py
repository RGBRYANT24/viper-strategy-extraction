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
import torch
from collections import deque
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gym_env
from gym_env.tictactoe_delta_selfplay import TicTacToeDeltaSelfPlayEnv
from gym_env.policies import RandomPlayerPolicy, MinMaxPlayerPolicy
from gymnasium import spaces


class PolicySnapshot:
    """
    策略快照包装器
    使用 state_dict 保存策略，避免 deepcopy 问题
    """
    def __init__(self, policy, device='cpu'):
        """
        Args:
            policy: MaskablePPO 的 policy 对象
            device: 运行设备
        """
        self.device = device
        # 保存策略的状态字典（克隆以确保独立性）
        # 对于小模型，可以直接保存在原设备上，但克隆到 CPU 更通用
        self.policy_state_dict = {k: v.cpu().clone() for k, v in policy.state_dict().items()}
        # 保存策略的类和参数（用于重建）
        self.policy_class = type(policy)
        self.observation_space = policy.observation_space
        self.action_space = policy.action_space

        # 保存网络架构参数（关键！）
        self.policy_kwargs = {}
        if hasattr(policy, 'features_extractor_class'):
            self.policy_kwargs['features_extractor_class'] = policy.features_extractor_class
        if hasattr(policy, 'features_extractor_kwargs'):
            self.policy_kwargs['features_extractor_kwargs'] = policy.features_extractor_kwargs
        if hasattr(policy, 'net_arch'):
            self.policy_kwargs['net_arch'] = policy.net_arch
        if hasattr(policy, 'activation_fn'):
            self.policy_kwargs['activation_fn'] = policy.activation_fn
        if hasattr(policy, 'ortho_init'):
            self.policy_kwargs['ortho_init'] = policy.ortho_init

        # 创建策略实例并加载状态
        self._policy = None
        self._create_policy()

    def _create_policy(self):
        """创建策略实例并加载状态"""
        # 创建新的策略实例，传递所有保存的参数
        self._policy = self.policy_class(
            observation_space=self.observation_space,
            action_space=self.action_space,
            lr_schedule=lambda _: 0.0,  # 不需要优化器，所以学习率为0
            **self.policy_kwargs  # 传递网络架构等参数
        )
        # 加载状态字典
        self._policy.load_state_dict(self.policy_state_dict)
        self._policy.to(self.device)
        self._policy.eval()  # 设置为评估模式

    def predict(self, observation, deterministic=False, action_masks=None):
        """
        预测动作（兼容 MaskablePPO 接口）

        Args:
            observation: 观察值
            deterministic: 是否使用确定性策略
            action_masks: 动作掩码 (可选)，shape=(n_actions,), 1=合法, 0=非法

        Returns:
            action: 动作
            state: 状态（用于兼容性，PPO不使用）
        """
        with torch.no_grad():
            # 将观察转换为张量
            if not isinstance(observation, torch.Tensor):
                obs_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
            else:
                obs_tensor = observation.unsqueeze(0).to(self.device)

            # 获取动作分布
            distribution = self._policy.get_distribution(obs_tensor)

            # 如果提供了 action_masks，应用 mask
            if action_masks is not None:
                # 将 mask 转换为张量
                if not isinstance(action_masks, torch.Tensor):
                    mask_tensor = torch.as_tensor(action_masks, dtype=torch.bool).unsqueeze(0).to(self.device)
                else:
                    mask_tensor = action_masks.unsqueeze(0).to(self.device)

                # 应用 mask：将非法动作的 logits 设为 -inf
                # 注意：distribution 是 Categorical，需要修改其 logits
                if hasattr(distribution.distribution, 'logits'):
                    logits = distribution.distribution.logits.clone()
                    logits[~mask_tensor] = -float('inf')
                    # 重新创建分布
                    from torch.distributions import Categorical
                    distribution.distribution = Categorical(logits=logits)

            # 采样动作
            if deterministic:
                action = distribution.mode()
            else:
                action = distribution.sample()

            return action.cpu().numpy()[0], None


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
    parser.add_argument("--ent-coef", type=float, default=0.05,
                       help="PPO 熵系数 (增加探索性)")
    parser.add_argument("--random-weight", type=float, default=2.0,
                       help="Random 对手的采样权重（相对于其他对手）")
    args = parser.parse_args()

    print("=" * 70)
    print(f"TicTacToe MaskablePPO Training (Delta-{args.max_pool_size}-Uniform)")
    print("=" * 70)
    print(f"总步数: {args.total_timesteps}")
    print(f"并行环境数: {args.n_env}")
    print(f"熵系数: {args.ent_coef} (控制探索性)")
    print(f"Random 对手权重: {args.random_weight}x")
    print(f"✓ 使用 Action Masking")
    print()

    # --- 步骤 1: 初始化策略池 ---
    temp_env = TicTacToeDeltaSelfPlayEnv()
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    temp_env.close()

    # 始终添加 Random 作为初始对手
    baseline_policies = [RandomPlayerPolicy(obs_space, act_space)]

    # 如果指定，添加 MinMax（但默认降低其采样权重）
    if args.use_minmax:
        baseline_policies.append(MinMaxPlayerPolicy(obs_space, act_space))

    learned_policy_pool = deque(maxlen=args.max_pool_size)

    print(f"基准策略池: {len(baseline_policies)} 个对手")
    for i, policy in enumerate(baseline_policies):
        print(f"  {i+1}. {policy.__class__.__name__}")
    print(f"学习策略池: 容量 = {args.max_pool_size}")
    print()

    # --- 步骤 2: 创建环境（带 Action Masking）---
    # 创建加权采样的环境包装器
    class WeightedSelfPlayEnv(TicTacToeDeltaSelfPlayEnv):
        """支持加权采样的自我对弈环境"""
        def __init__(self, random_weight=1.0, **kwargs):
            super().__init__(**kwargs)
            self.random_weight = random_weight

        def _sample_opponent(self):
            """加权采样对手"""
            all_opponents = []
            weights = []

            # 基准策略池（Random 有更高权重）
            for policy in self.baseline_pool:
                all_opponents.append(policy)
                if isinstance(policy, RandomPlayerPolicy):
                    weights.append(self.random_weight)  # Random 对手权重更高
                else:
                    weights.append(1.0)

            # 学习策略池（正常权重）
            if self.learned_pool is not None and len(self.learned_pool) > 0:
                for policy in list(self.learned_pool):
                    all_opponents.append(policy)
                    weights.append(1.0)

            if len(all_opponents) == 0:
                return None

            # 归一化权重并采样
            weights = np.array(weights)
            probs = weights / weights.sum()
            idx = np.random.choice(len(all_opponents), p=probs)

            return all_opponents[idx]

    def make_masked_env():
        env = WeightedSelfPlayEnv(
            baseline_pool=baseline_policies,
            learned_pool=learned_policy_pool,
            play_as_o_prob=args.play_as_o_prob,
            sampling_strategy='uniform',
            random_weight=args.random_weight  # 传递 Random 权重
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
        ent_coef=args.ent_coef,  # 熵系数，鼓励探索（从参数读取）
        policy_kwargs={'net_arch': net_arch},
        verbose=args.verbose,
        seed=args.seed
    )

    print("MaskablePPO 模型配置:")
    print(f"  网络结构: {net_arch}")
    print(f"  学习率: {model.learning_rate}")
    print(f"  批大小: {model.batch_size}")
    print(f"  熵系数: {model.ent_coef} (控制探索)")
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

        # 使用 PolicySnapshot 包装器保存策略（避免 deepcopy 问题）
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        current_policy_snapshot = PolicySnapshot(model.policy, device=device)
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
            # 获取 action mask 并传递给 predict
            action_mask = mask_fn(test_env)
            action, _ = model.predict(obs, deterministic=True, action_masks=action_mask)
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
