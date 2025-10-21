"""
VIPER with Regression Trees for TicTacToe
使用回归树替代分类树，输出Q值而不是单个动作

核心思想：
- 分类树输出：单个动作类别（0-8）→ 可能非法
- 回归树输出：9个Q值 → 可以用掩码过滤非法动作

优势：
1. 推理时可以应用合法动作掩码
2. 自然支持次优动作选择
3. Q值提供了动作价值的可解释性
"""

import warnings
import numpy as np
import torch
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from tqdm import tqdm

from gym_env import make_env
from model.paths import get_oracle_path, get_viper_path
from model.tree_wrapper import TreeWrapper
from test.evaluate import evaluate_policy
from train.oracle import get_model_cls
from train.viper import spaces_are_compatible


class QValueTreeWrapper:
    """
    回归树包装器，用于Q值预测

    与分类树的区别：
    - 分类树：输出单个动作
    - 回归树：输出9个Q值，然后选择合法动作中Q值最高的
    """

    def __init__(self, tree_model):
        """
        Args:
            tree_model: sklearn的DecisionTreeRegressor或MultiOutputRegressor
        """
        self.tree = tree_model
        self.n_actions = 9  # TicTacToe固定9个动作

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        """
        预测动作（兼容stable-baselines3接口）

        Args:
            observation: 棋盘状态 shape (n_env, 9) 或 (9,)

        Returns:
            actions: 预测的动作 shape (n_env,) 或标量
            state: 状态（None）
        """
        # 处理输入shape
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
            single_obs = True
        else:
            single_obs = False

        # 预测Q值 shape: (n_env, 9)
        q_values = self.tree.predict(observation)

        # 对每个环境选择Q值最高的合法动作
        actions = []
        for i in range(observation.shape[0]):
            obs = observation[i]
            q = q_values[i] if q_values.ndim > 1 else q_values

            # 获取合法动作（空位置）
            legal_actions = np.where(obs == 0)[0]

            if len(legal_actions) == 0:
                # 没有合法动作，返回任意动作（游戏应该已结束）
                action = 0
            else:
                # 选择合法动作中Q值最高的
                legal_q = q[legal_actions]
                best_idx = np.argmax(legal_q)
                action = legal_actions[best_idx]

            actions.append(action)

        actions = np.array(actions)

        if single_obs:
            return actions[0], None
        else:
            return actions, None

    def predict_q_values(self, observation):
        """
        直接返回Q值（用于分析）

        Args:
            observation: 棋盘状态

        Returns:
            q_values: 9个位置的Q值
        """
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        return self.tree.predict(observation)[0]

    def save(self, path):
        """保存模型"""
        import joblib
        joblib.dump(self.tree, path)
        print(f"Regression tree saved to: {path}")

    def print_info(self):
        """打印模型信息"""
        print("\n=== Regression Tree Model Info ===")

        # 如果是MultiOutputRegressor，获取第一个估计器
        if isinstance(self.tree, MultiOutputRegressor):
            base_tree = self.tree.estimators_[0]
            print(f"Model type: MultiOutputRegressor with {len(self.tree.estimators_)} trees")
        else:
            base_tree = self.tree
            print(f"Model type: Single DecisionTreeRegressor")

        if hasattr(base_tree, 'tree_'):
            print(f"Number of leaves: {base_tree.tree_.n_leaves}")
            print(f"Tree depth: {base_tree.tree_.max_depth}")
            print(f"Number of features: {base_tree.tree_.n_features}")

        print("="*40)


def extract_q_values_from_oracle(oracle, observations):
    """
    从Oracle（神经网络）中提取Q值

    Args:
        oracle: 训练好的DQN/PPO模型
        observations: 棋盘状态数组 (n_samples, 9)

    Returns:
        q_values: Q值数组 (n_samples, 9)
    """
    if isinstance(oracle, DQN):
        # DQN直接有Q网络
        obs_tensor = torch.FloatTensor(observations).to(oracle.device)
        with torch.no_grad():
            q_values = oracle.q_net(obs_tensor).cpu().numpy()
        return q_values

    elif isinstance(oracle, PPO):
        # PPO使用策略网络的log概率作为Q值的代理
        obs_tensor = torch.FloatTensor(observations).to(oracle.device)
        with torch.no_grad():
            # 获取所有动作的log概率
            features = oracle.policy.extract_features(obs_tensor)
            latent_pi = oracle.policy.mlp_extractor.forward_actor(features)
            action_logits = oracle.policy.action_net(latent_pi)

            # 使用softmax后的概率作为Q值的代理
            # 这不是真实的Q值，但可以反映动作的相对优劣
            q_values = torch.softmax(action_logits, dim=1).cpu().numpy()

        return q_values

    else:
        raise NotImplementedError(f"Oracle type {type(oracle)} not supported")


def train_viper_regression(args):
    """
    使用回归树训练VIPER

    主要改变：
    1. 使用DecisionTreeRegressor而不是DecisionTreeClassifier
    2. 训练目标是Q值而不是动作类别
    3. 支持MultiOutputRegressor用于多输出回归
    """
    print(f"Training VIPER with Regression Trees on {args.env_name}")
    print(f"Using Q-value regression approach\n")

    dataset = []
    policy = None
    policies = []
    rewards = []

    # 加载Oracle
    env, oracle = load_oracle_env(args)

    # 显示Oracle信息
    from model.paths import get_oracle_path
    oracle_path = get_oracle_path(args)
    print(f"✓ Loaded Oracle from: {oracle_path}")
    print(f"  Oracle type: {type(oracle).__name__}")
    if hasattr(oracle, 'policy'):
        print(f"  Policy type: {type(oracle.policy).__name__}")
    print()

    for iteration in tqdm(range(args.n_iter), disable=args.verbose > 0):
        # 第一次迭代使用Oracle，之后使用当前策略
        beta = 1 if iteration == 0 else 0
        new_data = sample_trajectory_with_qvalues(args, oracle, policy, beta)
        dataset += new_data

        if args.verbose == 2:
            print(f"\nIteration {iteration + 1}/{args.n_iter}")
            print(f"Dataset size: {len(dataset)}")

        # 准备训练数据
        X = np.array([traj[0] for traj in dataset])  # 状态
        Q = np.array([traj[1] for traj in dataset])  # Q值 (n_samples, 9)
        weights = np.array([traj[2] for traj in dataset])  # 权重

        # 创建回归树
        # 方案1: MultiOutputRegressor - 为每个输出训练一棵树
        base_tree = DecisionTreeRegressor(
            max_depth=args.max_depth,
            max_leaf_nodes=args.max_leaves,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )

        # 使用MultiOutputRegressor包装
        regressor = MultiOutputRegressor(base_tree, n_jobs=-1)

        # 训练（注意：MultiOutputRegressor不直接支持sample_weight，需要特殊处理）
        if args.verbose == 2:
            print(f"Training regression tree...")

        # 简单方法：使用权重复制样本（适用于小数据集）
        if len(dataset) < 10000:
            # 将权重归一化并转换为整数采样次数
            normalized_weights = weights / weights.min()
            sample_counts = np.round(normalized_weights).astype(int)

            # 复制样本
            X_weighted = np.repeat(X, sample_counts, axis=0)
            Q_weighted = np.repeat(Q, sample_counts, axis=0)

            regressor.fit(X_weighted, Q_weighted)
        else:
            # 对于大数据集，直接训练不加权
            regressor.fit(X, Q)

        # 包装模型
        policy = regressor
        policies.append(regressor)

        # 评估策略
        wrapped_policy = QValueTreeWrapper(policy)
        env_eval = make_env(args, test_viper=True)
        mean_reward, std_reward = evaluate_policy(wrapped_policy, env_eval, n_eval_episodes=100)

        if args.verbose >= 1:
            print(f"Iteration {iteration + 1}: Reward = {mean_reward:.4f} +/- {std_reward:.4f}")

        rewards.append(mean_reward)

    # 选择最佳策略
    print(f"\nVIPER Regression training complete. Dataset size: {len(dataset)}")
    best_idx = np.argmax(rewards)
    best_policy = policies[best_idx]

    print(f"Best policy: Iteration {best_idx + 1}")
    print(f"Mean reward: {np.max(rewards):.4f}")

    # 保存
    wrapper = QValueTreeWrapper(best_policy)
    wrapper.print_info()

    # 修改路径以区分回归树和分类树
    path = get_viper_path(args)
    path_regression = path.replace('.joblib', '_regression.joblib')
    wrapper.save(path_regression)

    return wrapper


def sample_trajectory_with_qvalues(args, oracle, policy, beta):
    """
    采样轨迹并提取Q值

    与原始VIPER的区别：
    - 不仅记录Oracle选择的动作
    - 记录所有动作的Q值

    Returns:
        trajectory: [(state, q_values, weight), ...]
    """
    env, oracle = load_oracle_env(args)
    policy = policy or oracle

    trajectory = []
    reset_result = env.reset()

    # 处理reset返回值（可能是tuple或直接是obs）
    if isinstance(reset_result, tuple):
        obs = reset_result[0]
    else:
        obs = reset_result

    # 如果是向量化环境，取第一个
    if isinstance(obs, np.ndarray) and obs.ndim > 1:
        obs = obs[0]

    n_steps = args.total_timesteps // args.n_iter
    collected_steps = 0

    while collected_steps < n_steps:
        # 选择使用哪个策略（beta概率使用policy，1-beta使用oracle）
        use_policy = np.random.binomial(1, beta) == 1

        if use_policy and policy is not None and not isinstance(policy, BaseAlgorithm):
            # 使用回归树策略
            wrapped_policy = QValueTreeWrapper(policy)
            action, _ = wrapped_policy.predict(obs)
        else:
            # 使用Oracle
            action, _ = oracle.predict(obs, deterministic=True)

        # 处理action可能是数组的情况（向量化环境）
        if isinstance(action, np.ndarray):
            if action.ndim > 0 and action.size > 0:
                action = action.item() if action.size == 1 else action[0]
            else:
                action = int(action)  # 0维数组转标量

        # 从Oracle提取Q值（这是我们的训练目标）
        # 处理obs可能是向量化的情况
        if isinstance(obs, np.ndarray):
            if obs.ndim == 1:
                # 单个环境
                obs_batch = obs.reshape(1, -1)
                obs_for_record = obs.copy()
            else:
                # 多个环境，取第一个
                obs_batch = obs[0:1]
                obs_for_record = obs[0].copy()
        else:
            obs_batch = obs
            obs_for_record = obs

        q_values = extract_q_values_from_oracle(oracle, obs_batch)[0]

        # 计算状态重要性权重
        # get_loss需要的是原始obs（可能是向量化的）
        state_loss = get_loss(env, oracle, obs_batch)
        if isinstance(state_loss, np.ndarray):
            state_loss = state_loss[0] if state_loss.size > 0 else 1.0

        # 记录 (state, q_values, weight)
        trajectory.append((obs_for_record,
                          q_values,
                          state_loss))

        # 执行动作
        step_result = env.step(action)
        if len(step_result) == 5:
            next_obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            next_obs, reward, done, info = step_result

        # 处理done可能是数组的情况（向量化环境）
        if isinstance(done, np.ndarray):
            done = done.any()  # 任何一个环境结束就重置

        # 更新观测
        if isinstance(next_obs, np.ndarray) and next_obs.ndim > 1:
            # 向量化环境，取第一个
            obs = next_obs[0]
        else:
            obs = next_obs

        collected_steps += 1

        if done:
            reset_obs = env.reset()
            # 处理reset返回的观测
            if isinstance(reset_obs, tuple):
                # 新版gym返回 (obs, info)
                obs = reset_obs[0]
            else:
                obs = reset_obs

            # 如果是向量化环境
            if isinstance(obs, np.ndarray) and obs.ndim > 1:
                obs = obs[0]

    return trajectory


def load_oracle_env(args):
    """加载Oracle环境（与原始VIPER相同）"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = make_env(args)
        model_cls, _ = get_model_cls(args)

        # Patch spaces check
        import stable_baselines3.common.base_class as base_class_module

        def patched_check(env, observation_space, action_space):
            if not spaces_are_compatible(observation_space, env.observation_space):
                raise ValueError(
                    f"Observation spaces do not match: {observation_space} != {env.observation_space}"
                )
            if not spaces_are_compatible(action_space, env.action_space):
                raise ValueError(
                    f"Action spaces do not match: {action_space} != {env.action_space}"
                )

        original_check = base_class_module.check_for_correct_spaces
        base_class_module.check_for_correct_spaces = patched_check

        try:
            oracle = model_cls.load(get_oracle_path(args), env=env)
            oracle.verbose = args.verbose
            env = oracle.env
            return env, oracle
        finally:
            base_class_module.check_for_correct_spaces = original_check


def get_loss(env, model: BaseAlgorithm, obs):
    """计算状态重要性（与原始VIPER相同）"""
    if isinstance(model, DQN):
        obs_tensor = torch.from_numpy(obs if isinstance(obs, np.ndarray) else np.array(obs))
        obs_tensor = obs_tensor.to(model.device)
        q_values = model.q_net(obs_tensor).detach().cpu().numpy()
        return q_values.max(axis=-1) - q_values.min(axis=-1)

    elif isinstance(model, PPO):
        import gym
        assert isinstance(env.action_space, gym.spaces.Discrete), \
            "Only discrete action spaces supported"

        possible_actions = np.arange(env.action_space.n)
        obs_tensor = torch.from_numpy(
            obs if isinstance(obs, np.ndarray) else np.array(obs)
        ).to(model.device)

        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        log_probs = []
        for action in possible_actions:
            action_tensor = torch.from_numpy(np.array([action])).repeat(
                obs_tensor.shape[0]
            ).to(model.device)
            _, log_prob, _ = model.policy.evaluate_actions(obs_tensor, action_tensor)
            log_probs.append(log_prob.detach().cpu().numpy().flatten())

        log_probs = np.array(log_probs).T
        return log_probs.max(axis=-1) - log_probs.min(axis=-1)

    raise NotImplementedError(f"Model type {type(model)} not supported")


if __name__ == "__main__":
    # 简单测试
    print("VIPER Regression Tree Training Module")
    print("Use main.py with --use-regression flag to train")
