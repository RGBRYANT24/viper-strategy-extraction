"""
VIPER with Single Classification Tree + Probability Masking
单棵分类树 + 概率掩码方案

核心思想：
- 保持单棵分类树（高可解释性）
- 使用 predict_proba() 获取概率分布
- 应用合法动作掩码选择最佳合法动作

优势：
1. ✅ 单棵树 - 完整的IF-THEN规则可解释性
2. ✅ 100%避免非法动作
3. ✅ 模型更小（vs 9棵回归树）
4. ✅ 可以提取condition-action pairs
"""

import warnings
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from gym_env import make_env
from model.paths import get_oracle_path, get_viper_path
from model.tree_wrapper import TreeWrapper
from test.evaluate import evaluate_policy
from train.oracle import get_model_cls
from train.viper import spaces_are_compatible, load_oracle_env, sample_trajectory


class ProbabilityMaskedTreeWrapper:
    """
    带概率掩码的分类树包装器

    核心特点：
    - 使用单棵分类树（predict_proba）
    - 推理时应用合法动作掩码
    - 保持完整的可解释性
    """

    def __init__(self, tree_model):
        """
        Args:
            tree_model: sklearn的DecisionTreeClassifier
        """
        self.tree = tree_model
        self.n_actions = 9  # TicTacToe固定9个动作

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        """
        预测动作（带合法动作掩码）

        Args:
            observation: 棋盘状态 shape (n_env, 9) 或 (9,)

        Returns:
            actions: 预测的动作
            state: None
        """
        # 处理输入shape
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
            single_obs = True
        else:
            single_obs = False

        # 获取概率分布 shape: (n_env, n_classes)
        action_probs = self.tree.predict_proba(observation)

        # 对每个环境选择最佳合法动作
        actions = []
        for i in range(observation.shape[0]):
            obs = observation[i]
            probs = action_probs[i]

            # 获取合法动作（空位置）
            legal_actions = np.where(obs == 0)[0]

            if len(legal_actions) == 0:
                # 没有合法动作（不应该发生）
                action = 0
            else:
                # 创建掩码概率
                masked_probs = np.full(self.n_actions, -np.inf)
                masked_probs[legal_actions] = probs[legal_actions]

                # 选择合法动作中概率最高的
                action = np.argmax(masked_probs)

            actions.append(action)

        actions = np.array(actions)

        if single_obs:
            return actions[0], None
        else:
            return actions, None

    def get_action_probabilities(self, observation):
        """
        获取动作概率（用于分析）

        Returns:
            probs: 9个动作的概率
        """
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        return self.tree.predict_proba(observation)[0]

    def save(self, path):
        """保存模型"""
        import joblib
        joblib.dump(self.tree, path)
        print(f"Classification tree saved to: {path}")

    def print_info(self):
        """打印模型信息"""
        print("\n=== Single Classification Tree with Probability Masking ===")
        print(f"Number of leaves: {self.tree.tree_.n_leaves}")
        print(f"Tree depth: {self.tree.tree_.max_depth}")
        print(f"Number of classes: {self.tree.n_classes_}")
        print(f"Classes: {self.tree.classes_}")
        print("="*60)


def train_viper_single_tree(args):
    """
    使用单棵分类树 + 概率掩码训练VIPER

    这个方案保持了完整的可解释性：
    - 单棵决策树
    - 可以提取完整的IF-THEN规则
    - 可以可视化整个决策过程
    """
    print(f"Training VIPER with Single Classification Tree + Probability Masking")
    print(f"Environment: {args.env_name}\n")

    dataset = []
    policy = None
    policies = []
    rewards = []

    for iteration in tqdm(range(args.n_iter), disable=args.verbose > 0):
        beta = 1 if iteration == 0 else 0

        # 使用原始VIPER的sample_trajectory
        dataset += sample_trajectory(args, policy, beta)

        if args.verbose == 2:
            print(f"\nIteration {iteration + 1}/{args.n_iter}")
            print(f"Dataset size: {len(dataset)}")

        # 准备训练数据
        X = np.array([traj[0] for traj in dataset])  # 状态
        y = np.array([traj[1] for traj in dataset])  # 动作标签
        weights = np.array([traj[2] for traj in dataset])  # 权重

        # 创建分类树（与原始VIPER相同）
        clf = DecisionTreeClassifier(
            ccp_alpha=0.0001,
            criterion="entropy",
            max_depth=args.max_depth,
            max_leaf_nodes=args.max_leaves,
            random_state=42
        )

        # 训练
        clf.fit(X, y, sample_weight=weights)

        # 包装模型（添加概率掩码功能）
        policy = clf
        policies.append(clf)

        # 评估策略
        wrapped_policy = ProbabilityMaskedTreeWrapper(policy)
        env_eval = make_env(args, test_viper=True)
        mean_reward, std_reward = evaluate_policy(wrapped_policy, env_eval, n_eval_episodes=100)

        if args.verbose >= 1:
            print(f"Iteration {iteration + 1}: Reward = {mean_reward:.4f} +/- {std_reward:.4f}")

        rewards.append(mean_reward)

    # 选择最佳策略
    print(f"\nVIPER Single Tree training complete. Dataset size: {len(dataset)}")
    best_idx = np.argmax(rewards)
    best_policy = policies[best_idx]

    print(f"Best policy: Iteration {best_idx + 1}")
    print(f"Mean reward: {np.max(rewards):.4f}")

    # 保存
    wrapper = ProbabilityMaskedTreeWrapper(best_policy)
    wrapper.print_info()

    # 修改路径以区分
    path = get_viper_path(args)
    path_single = path.replace('.joblib', '_single_tree.joblib')
    wrapper.save(path_single)

    return wrapper


if __name__ == "__main__":
    print("VIPER Single Tree + Probability Masking Training Module")
    print("Use main.py with train-viper-single command to train")
