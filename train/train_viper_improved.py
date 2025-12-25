"""
改进的VIPER训练策略 - 解决决策树过拟合/欠拟合问题

核心改进:
1. 使用混合对手策略：神经网络自身 + 随机探索
2. 添加数据增强：通过对称性扩展数据集
3. 动态调整采样策略：平衡exploitation和exploration
4. 加入状态覆盖度监控
"""

import warnings
import argparse
import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from tqdm import tqdm
import joblib
from collections import defaultdict

from gym_env import make_env
from model.paths import get_oracle_path, get_viper_path
from model.tree_wrapper import TreeWrapper
from test.evaluate import evaluate_policy
from train.oracle import get_model_cls


def get_state_hash(obs):
    """计算状态的哈希值，用于追踪状态覆盖度"""
    return tuple(obs.flatten())


def get_symmetric_states_tictactoe(board):
    """
    生成井字棋的所有对称状态
    返回: 列表，包含所有对称变换后的(状态, 动作映射)
    """
    board_2d = board.reshape(3, 3)
    symmetries = []

    # 定义8种对称变换
    transformations = [
        ('identity', lambda b: b, lambda a: a),
        ('rot90', lambda b: np.rot90(b, k=1), lambda a: rotate_action_90(a)),
        ('rot180', lambda b: np.rot90(b, k=2), lambda a: rotate_action_180(a)),
        ('rot270', lambda b: np.rot90(b, k=3), lambda a: rotate_action_270(a)),
        ('flip_h', lambda b: np.fliplr(b), lambda a: flip_action_h(a)),
        ('flip_v', lambda b: np.flipud(b), lambda a: flip_action_v(a)),
        ('flip_d1', lambda b: np.transpose(b), lambda a: flip_action_d1(a)),
        ('flip_d2', lambda b: np.fliplr(np.transpose(b)), lambda a: flip_action_d2(a)),
    ]

    for name, state_transform, action_transform in transformations:
        transformed_board = state_transform(board_2d).flatten()
        symmetries.append((transformed_board, action_transform))

    return symmetries


def rotate_action_90(action):
    """将动作按顺时针90度旋转对应的映射"""
    # 0 1 2    6 3 0
    # 3 4 5 -> 7 4 1
    # 6 7 8    8 5 2
    mapping = {0: 6, 1: 3, 2: 0, 3: 7, 4: 4, 5: 1, 6: 8, 7: 5, 8: 2}
    return mapping[action]


def rotate_action_180(action):
    """将动作按180度旋转"""
    mapping = {0: 8, 1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1, 8: 0}
    return mapping[action]


def rotate_action_270(action):
    """将动作按逆时针90度旋转"""
    mapping = {0: 2, 1: 5, 2: 8, 3: 1, 4: 4, 5: 7, 6: 0, 7: 3, 8: 6}
    return mapping[action]


def flip_action_h(action):
    """水平翻转动作"""
    mapping = {0: 2, 1: 1, 2: 0, 3: 5, 4: 4, 5: 3, 6: 8, 7: 7, 8: 6}
    return mapping[action]


def flip_action_v(action):
    """垂直翻转动作"""
    mapping = {0: 6, 1: 7, 2: 8, 3: 3, 4: 4, 5: 5, 6: 0, 7: 1, 8: 2}
    return mapping[action]


def flip_action_d1(action):
    """主对角线翻转"""
    mapping = {0: 0, 1: 3, 2: 6, 3: 1, 4: 4, 5: 7, 6: 2, 7: 5, 8: 8}
    return mapping[action]


def flip_action_d2(action):
    """副对角线翻转"""
    mapping = {0: 8, 1: 5, 2: 2, 3: 7, 4: 4, 5: 1, 6: 6, 7: 3, 8: 0}
    return mapping[action]


def augment_trajectory_with_symmetry(trajectory, env_name):
    """
    使用对称性扩展训练数据

    Args:
        trajectory: [(obs, action, weight), ...]
        env_name: 环境名称

    Returns:
        扩展后的trajectory
    """
    if 'TicTacToe' not in env_name:
        return trajectory  # 只对井字棋做对称性扩展

    augmented = []
    for obs, action, weight in trajectory:
        # 添加原始数据
        augmented.append((obs, action, weight))

        # 添加对称变换的数据
        symmetries = get_symmetric_states_tictactoe(obs)
        for transformed_obs, action_transform in symmetries[1:]:  # 跳过identity
            transformed_action = action_transform(action)
            augmented.append((transformed_obs, transformed_action, weight * 0.5))  # 降低权重

    return augmented


def sample_trajectory_improved(args, policy, oracle, beta,
                               epsilon_random=0.0, use_augmentation=True):
    """
    改进的轨迹采样策略

    Args:
        args: 参数配置
        policy: 当前决策树策略 (或None)
        oracle: 神经网络Oracle
        beta: 使用Oracle的概率 (1-beta使用policy)
        epsilon_random: 添加随机探索的概率
        use_augmentation: 是否使用数据增强

    Returns:
        trajectory: [(obs, action, weight), ...]
    """
    from train.viper import load_oracle_env, get_loss

    env, oracle = load_oracle_env(args)
    policy = policy or oracle

    trajectory = []
    state_coverage = set()  # 追踪访问过的状态

    obs = env.reset()
    n_steps = args.total_timesteps // args.n_iter
    step_count = 0

    # 检测是否为向量化环境
    is_vectorized = len(obs.shape) > 1 and obs.shape[0] > 1
    n_envs = obs.shape[0] if is_vectorized else 1

    while step_count < n_steps:
        # 策略选择：Oracle, policy, 或随机
        rand = np.random.random()

        if rand < epsilon_random:
            # 随机探索
            if is_vectorized:
                action = np.array([np.random.randint(0, env.action_space.n) for _ in range(n_envs)])
            else:
                legal_actions = np.where(obs == 0)[0]
                action = np.random.choice(legal_actions) if len(legal_actions) > 0 else np.random.randint(0, env.action_space.n)
            active_policy = 'random'
        elif rand < epsilon_random + beta:
            # 使用Oracle
            action, _ = oracle.predict(obs, deterministic=True)
            active_policy = 'oracle'
        else:
            # 使用当前policy (决策树)
            if isinstance(policy, DecisionTreeClassifier):
                if is_vectorized:
                    # 向量化环境：为每个环境预测动作
                    action = policy.predict(obs)
                else:
                    # 单个环境
                    action = policy.predict(obs.reshape(1, -1))[0]
            else:
                action, _ = policy.predict(obs, deterministic=True)
            active_policy = 'policy'

        # 获取Oracle的动作(作为标签)
        oracle_action, _ = oracle.predict(obs, deterministic=True)
        # 注意：不要在这里取[0]，因为向量化环境需要完整的动作数组

        # 执行动作
        step_result = env.step(action)
        if len(step_result) == 5:
            next_obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            next_obs, reward, done, info = step_result

        # 计算状态的重要性权重
        state_loss = get_loss(env, oracle, obs)

        # 处理向量化环境（使用前面已经检测的is_vectorized）
        if is_vectorized:
            # 向量化环境：处理多个环境
            n_envs = obs.shape[0]
            for i in range(n_envs):
                # 记录状态覆盖度
                state_hash = get_state_hash(obs[i])
                state_coverage.add(state_hash)

                # 添加到轨迹
                trajectory.append((
                    obs[i],
                    oracle_action[i] if hasattr(oracle_action, '__len__') and len(oracle_action) > 1 else oracle_action,
                    state_loss[i] if hasattr(state_loss, '__len__') and len(state_loss) > 1 else state_loss
                ))

                step_count += 1

            # 检查是否有环境完成（向量化环境的done是数组）
            if hasattr(done, '__len__') and np.any(done):
                obs = env.reset()
            else:
                obs = next_obs
        else:
            # 单个环境
            state_hash = get_state_hash(obs)
            state_coverage.add(state_hash)

            trajectory.append((obs, oracle_action, state_loss[0] if hasattr(state_loss, '__len__') else state_loss))

            obs = next_obs
            step_count += 1

            # 单个环境的done是布尔值
            if done:
                obs = env.reset()

    # 数据增强
    if use_augmentation and 'TicTacToe' in args.env_name:
        print(f"  原始数据: {len(trajectory)} 样本, 覆盖 {len(state_coverage)} 个状态")
        trajectory = augment_trajectory_with_symmetry(trajectory, args.env_name)
        print(f"  增强后: {len(trajectory)} 样本")

    return trajectory


def train_viper_improved(args):
    """改进的VIPER训练流程"""
    print(f"训练改进版VIPER: {args.env_name}")
    print(f"关键参数:")
    print(f"  - 迭代次数: {args.n_iter}")
    print(f"  - 最大叶子数: {args.max_leaves}")
    print(f"  - 最大深度: {args.max_depth}")
    print(f"  - 数据增强: {args.use_augmentation}")
    print(f"  - 探索策略: {args.exploration_strategy}")
    print()

    dataset = []
    policy = None
    policies = []
    rewards = []

    # 动态调整探索率
    for i in tqdm(range(args.n_iter), disable=args.verbose > 0):
        # 第一次迭代：完全使用Oracle
        if i == 0:
            beta = 1.0
            epsilon_random = 0.0
        else:
            # 逐渐减少Oracle使用，增加探索
            if args.exploration_strategy == 'decay':
                beta = max(0.1, 1.0 - i / args.n_iter)  # 从1.0衰减到0.1
                epsilon_random = min(0.3, i / args.n_iter * 0.3)  # 从0增加到0.3
            elif args.exploration_strategy == 'constant':
                beta = 0.5
                epsilon_random = 0.2
            else:  # original
                beta = 0.0
                epsilon_random = 0.0

        if args.verbose >= 1:
            print(f"\n--- 迭代 {i+1}/{args.n_iter} ---")
            print(f"  beta={beta:.2f}, epsilon_random={epsilon_random:.2f}")

        # 采样轨迹
        from train.viper import load_oracle_env, get_loss
        env, oracle = load_oracle_env(args)

        new_trajectory = sample_trajectory_improved(
            args, policy, oracle, beta,
            epsilon_random=epsilon_random,
            use_augmentation=args.use_augmentation
        )
        dataset += new_trajectory

        # 训练决策树
        clf = DecisionTreeClassifier(
            ccp_alpha=args.ccp_alpha,
            criterion="entropy",
            max_depth=args.max_depth,
            max_leaf_nodes=args.max_leaves,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf
        )

        x = np.array([traj[0] for traj in dataset])
        y = np.array([traj[1] for traj in dataset])
        weight = np.array([traj[2] for traj in dataset])

        # 归一化权重
        if weight.sum() > 0:
            weight = weight / weight.sum() * len(weight)

        clf.fit(x, y, sample_weight=weight)

        policies.append(clf)
        policy = clf

        # 评估策略
        env = make_env(args, test_viper=True)
        mean_reward, std_reward = evaluate_policy(TreeWrapper(policy), env, n_eval_episodes=100)
        rewards.append(mean_reward)

        if args.verbose >= 1:
            print(f"  数据集大小: {len(dataset)}")
            print(f"  叶子节点数: {policy.get_n_leaves()}")
            print(f"  树深度: {policy.get_depth()}")
            print(f"  评估得分: {mean_reward:.4f} ± {std_reward:.4f}")

    # 选择最佳策略
    print(f"\nVIPER训练完成！")
    print(f"  总数据量: {len(dataset)}")

    best_idx = np.argmax(rewards)
    best_policy = policies[best_idx]
    best_reward = rewards[best_idx]

    print(f"  最佳策略: 迭代 {best_idx + 1}")
    print(f"  最佳得分: {best_reward:.4f}")

    # 保存模型
    wrapper = TreeWrapper(best_policy)
    wrapper.print_info()

    path = get_viper_path(args)
    wrapper.save(path)
    print(f"\n模型已保存至: {path}")

    return wrapper


def main():
    parser = argparse.ArgumentParser(description="改进的VIPER训练")

    # 环境参数
    parser.add_argument("--env-name", type=str, default="TicTacToe-v0",
                       help="环境名称")
    parser.add_argument("--n-env", type=int, default=8,
                       help="并行环境数")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--verbose", type=int, default=1,
                       help="详细程度 (0=静默, 1=info, 2=debug)")
    parser.add_argument("--render", action='store_true',
                       help="是否渲染环境")

    # VIPER参数
    parser.add_argument("--n-iter", type=int, default=80,
                       help="VIPER迭代次数")
    parser.add_argument("--total-timesteps", type=int, default=100000,
                       help="总采样步数")
    parser.add_argument("--max-leaves", type=int, default=None,
                       help="决策树最大叶子数")
    parser.add_argument("--max-depth", type=int, default=None,
                       help="决策树最大深度")

    # 改进策略参数
    parser.add_argument("--use-augmentation", action='store_true', default=True,
                       help="使用数据增强(对称性)")
    parser.add_argument("--exploration-strategy", type=str, default='decay',
                       choices=['original', 'decay', 'constant'],
                       help="探索策略: original=标准VIPER, decay=渐进探索, constant=固定探索")
    parser.add_argument("--ccp-alpha", type=float, default=0.0001,
                       help="决策树复杂度惩罚")
    parser.add_argument("--min-samples-split", type=int, default=2,
                       help="分裂内部节点所需最小样本数")
    parser.add_argument("--min-samples-leaf", type=int, default=1,
                       help="叶子节点所需最小样本数")

    # 模型路径
    parser.add_argument("--oracle-path", type=str, default=None,
                       help="Oracle模型路径")
    parser.add_argument("--tictactoe-opponent", type=str, default="selfplay",
                       choices=['random', 'minmax', 'selfplay'],
                       help="井字棋对手类型 (仅影响路径命名)")
    parser.add_argument("--log-prefix", type=str, default="",
                       help="日志前缀")

    args = parser.parse_args()

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("\n" + "="*70)
    print("改进版 VIPER 训练")
    print("="*70)

    # 训练
    train_viper_improved(args)

    print("\n" + "="*70)
    print("训练完成！")
    print("="*70)


if __name__ == "__main__":
    main()
