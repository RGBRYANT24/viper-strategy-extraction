import numpy as np
import torch
import gymnasium as gym
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sb3_contrib import MaskablePPO
import joblib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_env.tictactoe import TicTacToeEnv

def test_env():
    env = gym.make('TicTacToe-v0', opponent_type='random')
    obs,_ = env.reset()
    print('Initial Observation:\n', obs.reshape(3,-1))

    action = 4
    obs, reward, done, truncated, info = env.step(action)
    print(f'\nAfter taking action {action}:')
    print('Observation:\n', obs.reshape(3,-1))
    print('Reward:', reward)
    print('Done:', done)
    print('Truncated:', truncated)
    print('Info:', info)
    return env, obs


def load_oracle(oracle_path):
    env = gym.make('TicTacToe-v0', opponent_type='random')
    oracle = MaskablePPO.load(oracle_path, env=env)
    obs, _ = env.reset()
    mask = (obs == 0).astype(bool)
    print('mask', mask)
    mask_tensor = torch.tensor(mask).unsqueeze(0)
    print('mask_tensor', mask_tensor)
    action, _ = oracle.predict(obs,deterministic=True, action_masks=mask_tensor)
    print('Predicted action by oracle:', action)
    return oracle

def sample_trajectory(orcale, env, n_steps, use_oracle=True):
    """采样 n_steps 个样本

    Returns:
        dataset: List of (observation, action, weight)
    """
    pass


def compute_criticality(env, model, observation):
    if isinstance(model, MaskablePPO):
        # For policy gradient methods we use the max entropy formulation
        # to get Q(s, a) \approx log pi(a|s)
        # See Ziebart et al. 2008
        assert isinstance(env.action_space,
                          gym.spaces.Discrete), "Only discrete action spaces supported for loss function"
        possible_actions = np.arange(env.action_space.n)
        mask = (observation == 0).astype(bool)
        mask_tensor = torch.tensor(mask).unsqueeze(0)
        # print('mask_tensor', mask_tensor)
        possible_actions = possible_actions[mask]
        # print('possible_actions after masking', possible_actions)
        obs_tensor = torch.as_tensor(observation).unsqueeze(0).to(model.device)
        # print('obs_tensor', obs_tensor)

        # MaskablePPO's evaluate_actions expects:
        # - obs: torch.Tensor (not a dict)
        # - actions: torch.Tensor
        # - action_masks: Optional[torch.Tensor] (as third parameter)

        # print("--- 检查张量 ---")
        # print("观测张量:", obs_tensor)
        # print("Mask 张量:", mask_tensor)
        # print("-----------------")

        log_probs = []
        for action in possible_actions:
            action_tensor = torch.tensor([action]).to(model.device)
            # print('action_tensor', action_tensor)
            _, log_prob, _ = model.policy.evaluate_actions(obs_tensor, action_tensor, action_masks=mask_tensor)
            log_probs.append(log_prob.detach().cpu().numpy().flatten())
            # print('log_prob for action', action, ':', log_prob.detach().cpu().numpy().flatten())

        log_probs = np.array(log_probs).T
        # print('log_probs array:', log_probs)
        # print('Max log prob:', log_probs.max(axis=1))
        # print('Min log prob:', log_probs.min(axis=1))
        # print('Criticality (max - min):', log_probs.max(axis=1) - log_probs.min(axis=1))
        return log_probs.max(axis=1) - log_probs.min(axis=1)
    
    # with torch.no_grad():
        #     # extract_features 只需要观察张量
        #     features = model.policy.extract_features(obs_tensor)
            
        #     # 获取潜在向量
        #     latent_pi, latent_vf = model.policy.mlp_extractor(features)
            
        #     # 获取动作分布
        #     distribution = model.policy._get_action_dist_from_latent(latent_pi)

        #     logits = model.policy.action_net(latent_pi)  # [batch, num_actions]
        #     print("原始 logits:", logits)

        #     # 应用掩码：将不可行的动作设置为极小值
        #     masked_logits = logits.clone()
        #     masked_logits[~mask_tensor] = -1e8
        #     print("掩码后的 logits:", masked_logits)

        #     log_probs = torch.log_softmax(masked_logits, dim=-1)
        #     print("掩码后的 log 概率:", log_probs)

        #     max_log_prob = torch.max(log_probs, dim=1)[0]
        #     print("最大 log 概率:", max_log_prob.cpu().numpy())
            
            # 获取所有动作的概率
            # all_probs = distribution.distribution.probs
            # print("所有动作的 (Masked) 概率:", all_probs.cpu().numpy())

        # log_probs = []
        # for action in possible_actions:
        #     action_te

        # with torch.no_grad():
        #     # 将观察转换为张量
        #     if not isinstance(observation, torch.Tensor):
        #         obs_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        #     else:
        #         obs_tensor = observation.unsqueeze(0).to(self.device)
        
    

    

def test_oracle():
    env = gym.make('TicTacToe-v0', opponent_type='random')
    oracle = MaskablePPO.load("log/oracle_TicTacToe_ppo_aggressive.zip", env=env)
    obs, _ = env.reset()
    done = False
    episode_reward = 0
    while not done:
        mask = (obs == 0).astype(bool)
        print('mask', mask, 'mask_shape', mask.shape)

        action,_ = oracle.predict(obs, deterministic=True, action_masks=torch.tensor(mask).unsqueeze(0))

        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        print('truncated', truncated)
        done = done or truncated

        print('Action taken:', action, 'reward:', reward)

    print('Episode reward:', episode_reward)
    env.close()


def get_oracle_action_logits(oracle, obs):
    """获取原始logits（不要mask！）"""
    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        obs_tensor = obs_tensor.to(oracle.device)

        distribution = oracle.policy.get_distribution(obs_tensor)
        logits = distribution.distribution.logits.cpu().numpy()[0]

        # ✅ 直接返回原始logits
        return logits

# Todo: 先后手采样没有调整

def sample_trajectory_regression(oracle, policy, env, n_steps, beta=1.0):
    dataset = []
    trajectory = []

    obs, _ = env.reset()
    policy = policy or oracle

    while len(trajectory) < n_steps:
        # print('Current observation:\n', obs.reshape(3,-1))

        # mixed strategy
        use_oracle_for_action = np.random.binomial(1, beta) == 1
        # use_oracle_for_action = True
        active_policy = oracle if use_oracle_for_action else policy
        action = None

        # mask
        mask = (obs == 0).astype(bool)

        # 获取神经网络的输出 logits
        oracle_logits = get_oracle_action_logits(oracle, obs)
        # oracle_logits = logits.clone()
        # print('oracle_logits', oracle_logits)
        masked_logits = np.full(9, -np.inf)

        # 应用掩码：将不可行的动作设置为极小值

        if isinstance(active_policy, MaskablePPO):
            # MaskablePPO
            masked_logits[mask] = oracle_logits[mask]
            action = np.argmax(masked_logits)
            # print('Action chosen by MaskablePPO:', action)
        else:
            # Decision Tree - predict() 返回 (action, state) 元组
            action, _ = active_policy.predict(obs.reshape(1,-1))

        # 采取行动并获取下一个状态
        # 使用active_policy采取行动 不是oracle
        step_result = env.step(action)
        if len(step_result) == 5:
            next_obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            next_obs, reward, done, info = step_result

        state_loss = compute_criticality(env, oracle, obs)

        # # 将 logits tensor 转换为 numpy 数组
        # oracle_logits = logits.squeeze().detach().cpu().numpy()  # shape: (num_actions,)
        # 提取权重标量
        weight = state_loss[0] if isinstance(state_loss, np.ndarray) else state_loss

        # print('observation:', obs)
        # print('oracle_logits shape:', oracle_logits.shape, 'values:', oracle_logits)
        # print('weight:', weight)
        # print('trajectory this time:', (obs.copy(), oracle_logits, weight))

        # 存储 (观察向量, logits向量, 权重标量) 用于回归树训练
        trajectory.append((obs.copy(), oracle_logits, weight))

        obs = next_obs

        if done:
            obs, _ = env.reset()

    # print('Sampled trajectory:', trajectory)

    return trajectory

def train_regression_tree(trajectory, max_depth=10, max_leaves=50):
    """训练回归决策树

    Args:
        trajectory: List of (obs, logits, weight)

    Returns:
        tree: DecisionTreeRegressor
    """

    # 准备训练数据
    X = np.array([obs for obs, _, _ in trajectory])       # (N, 9)
    y = np.array([logits for _, logits, _ in trajectory]) # (N, 9)
    weights = np.array([w for _, _, w in trajectory])     # (N,) 

    print(f"训练数据: X.shape={X.shape}, y.shape={y.shape}")

    tree = DecisionTreeRegressor(
        max_depth=max_depth,
        max_leaf_nodes=max_leaves,
        random_state=42,
        min_samples_split=10,  # 防止过拟合
        min_samples_leaf=5
    )

    tree.fit(X, y, sample_weight=weights)

    print(f"✓ 训练完成")
    print(f"  树深度: {tree.tree_.max_depth}")
    print(f"  叶子节点数: {tree.tree_.n_leaves}")

    return tree


class RegressionTreePolicy:
    """回归树策略包装器"""

    def __init__(self, tree):
        self.tree = tree
        self.n_actions = 9

    def predict(self, observation, deterministic=True):
        """预测动作（结合masking）

        Returns:
            action: 最优的合法动作
        """
        # 处理输入
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
            single_obs = True
        else:
            single_obs = False

        actions = []
        for obs in observation:
            # 预测所有动作的logits
            logits = self.tree.predict(obs.reshape(1, -1))[0]  # (9,)

            # 获取合法动作
            mask = (obs == 0).astype(bool)
            legal_actions = np.where(mask)[0]

            if len(legal_actions) == 0:
                # 无合法动作（不应发生）
                # actions.append(0)
                raise ValueError("No legal actions available!")
                continue

            # 选择logit最高的合法动作
            legal_logits = logits[legal_actions]
            best_idx = np.argmax(legal_logits)
            action = legal_actions[best_idx]

            actions.append(action)

        actions = np.array(actions)

        if single_obs:
            return actions[0], None
        else:
            return actions, None

    def get_action_logits(self, observation):
        """获取所有动作的logits（用于分析）

        Returns:
            logits: shape (9,)
        """
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)

        logits = self.tree.predict(observation)[0]
        return logits

def evaluate_policy(policy, env_name='TicTacToe-v0',
                   opponent_type='random', n_episodes=5000):
    """评估策略

    Returns:
        结果字典，包含 mean_reward, win_rate 等
    """
    env = gym.make(env_name, opponent_type=opponent_type)

    episode_rewards = []
    wins, draws, losses = 0, 0, 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            # 使用策略选择动作
            action, _ = policy.predict(obs, deterministic=True)

            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        episode_rewards.append(episode_reward)

        # 统计胜负平
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1

    env.close()

    # 计算统计量
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'win_rate': wins / n_episodes,
        'draw_rate': draws / n_episodes,
        'loss_rate': losses / n_episodes,
        'wins': wins,
        'draws': draws,
        'losses': losses
    }

def test_evaluation():
    """测试评估"""
    env = gym.make('TicTacToe-v0', opponent_type='random')
    oracle = MaskablePPO.load('log/oracle_TicTacToe_ppo_aggressive.zip', env=env)

    # 训练一个树
    # dataset = sample_trajectory_regression(oracle, env, n_steps=5000, use_oracle=True)
    dataset = sample_trajectory_regression(oracle, [], env, n_steps=5000, beta=1.0, use_criticality=True)
    tree = train_regression_tree(dataset, max_depth=10, max_leaves=50)
    policy = RegressionTreePolicy(tree)

    # 评估
    print("\n评估结果:")
    for opponent in ['random', 'minmax']:
        results = evaluate_policy(policy, opponent_type=opponent, n_episodes=5)
        print(f"\nvs {opponent.upper()}:")
        print(f"  胜率: {results['win_rate']*100:.1f}%")
        print(f"  平局率: {results['draw_rate']*100:.1f}%")
        print(f"  负率: {results['loss_rate']*100:.1f}%")
        print(f"  平均奖励: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")


def train_viper(oracle_path, output_path,
                n_iterations=10, samples_per_iter=5000,
                max_depth=10, max_leaves=50):
    """训练VIPER回归树策略"""
    from datetime import datetime
    import os

    # 创建带时间戳和参数的输出路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    params_str = f"iter{n_iterations}_samples{samples_per_iter}_depth{max_depth}_leaves{max_leaves}"

    # 创建专门的日志文件夹
    log_dir = "log/viper_mask_ppo_tictactoe"
    os.makedirs(log_dir, exist_ok=True)

    # 生成完整的输出路径
    if output_path:
        # 如果用户指定了路径，使用用户路径但添加时间戳和参数
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        output_path = os.path.join(log_dir, f"{base_name}_{timestamp}_{params_str}.joblib")
    else:
        output_path = os.path.join(log_dir, f"viper_tree_{timestamp}_{params_str}.joblib")

    print(f"模型将保存到: {output_path}\n")

    # 创建训练日志文件
    log_file = output_path.replace('.joblib', '_training_log.txt')
    with open(log_file, 'w') as f:
        f.write(f"VIPER Training Log\n")
        f.write(f"==================\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Oracle: {oracle_path}\n")
        f.write(f"Iterations: {n_iterations}\n")
        f.write(f"Samples per iter: {samples_per_iter}\n")
        f.write(f"Max depth: {max_depth}\n")
        f.write(f"Max leaves: {max_leaves}\n")
        f.write(f"Output: {output_path}\n")
        f.write(f"==================\n\n")

    # 1. 加载oracle
    env = gym.make('TicTacToe-v0', opponent_type='random')
    oracle = MaskablePPO.load(oracle_path, env=env)

    # 2.初始化
    all_data = []      # 累积所有数据（DAgger 的核心）
    all_trees = []     # 保存所有训练的树
    all_rewards = []   # 记录每棵树的性能
    policy = None      # 当前策略（初始为空）
    policies = []

    # 3. 迭代训练
    for iteration in range(n_iterations):
        print(f"\n{'='*70}")
        print(f"Iteration {iteration + 1}/{n_iterations}")
        print(f"{'='*70}")

        # 3.1 采样轨迹
        use_oracle = True
        beta = 1.0 if iteration == 0 else 0.0
        new_data = sample_trajectory_regression(oracle, policy, env, samples_per_iter, beta)

        # 3.2 累积数据
        all_data.extend(new_data)

        # 3.3 训练回归树
        tree = train_regression_tree(new_data, max_depth=max_depth, max_leaves=max_leaves)
        all_trees.append(tree)
        policy = RegressionTreePolicy(tree)
        policies.append(policy)

        # 3.4 评估新策略
        results = evaluate_policy(policy, opponent_type='random', n_episodes=100)
        all_rewards.append(results['mean_reward'])

        print(f"✓ 评估结果:")
        print(f"  平均奖励: {results['mean_reward']:.3f}")
        print(f"  胜: {results['wins']}, 平: {results['draws']}, 负: {results['losses']}")

        # 记录到日志文件
        with open(log_file, 'a') as f:
            f.write(f"Iteration {iteration + 1}/{n_iterations}\n")
            f.write(f"  Mean reward: {results['mean_reward']:.3f}\n")
            f.write(f"  Wins: {results['wins']}, Draws: {results['draws']}, Losses: {results['losses']}\n")
            f.write(f"  Win rate: {results['win_rate']*100:.1f}%\n")
            f.write(f"  Dataset size: {len(all_data)}\n")
            f.write("\n")

    # 4: 选择最佳树
    print("\n" + "="*70)
    print("VIPER 训练完成")
    print("="*70)

    best_idx = np.argmax(all_rewards)
    best_tree = all_trees[best_idx]
    best_reward = all_rewards[best_idx]

    print(f"最佳树: Iteration {best_idx + 1}")
    print(f"最佳奖励: {best_reward:.3f}")

    best_policy = RegressionTreePolicy(best_tree)
    joblib.dump(best_tree, output_path)
    print(f"✓ 模型保存到: {output_path}")

    # 记录最终结果到日志
    with open(log_file, 'a') as f:
        f.write("="*70 + "\n")
        f.write("Training Complete\n")
        f.write("="*70 + "\n")
        f.write(f"Best tree: Iteration {best_idx + 1}\n")
        f.write(f"Best reward: {best_reward:.3f}\n")
        f.write(f"Model saved to: {output_path}\n")
        f.write(f"Log saved to: {log_file}\n")

    print(f"✓ 训练日志保存到: {log_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train VIPER with MaskablePPO')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['test_env', 'test_oracle', 'test_evaluation', 'train'],
                       help='运行模式: test_env, test_oracle, test_evaluation, train')
    parser.add_argument('--oracle_path', type=str,
                       default='log/oracle_TicTacToe_ppo_aggressive.zip',
                       help='Oracle模型路径')
    parser.add_argument('--output_path', type=str,
                       default='viper_mask_ppo_tree',
                       help='输出决策树模型名称（不含扩展名，会自动添加时间戳和参数）')
    parser.add_argument('--n_iterations', type=int, default=10,
                       help='VIPER迭代次数')
    parser.add_argument('--samples_per_iter', type=int, default=50000,
                       help='每轮采样数量')
    parser.add_argument('--max_depth', type=int, default=10,
                       help='决策树最大深度')
    parser.add_argument('--max_leaves', type=int, default=50,
                       help='决策树最大叶子节点数')

    args = parser.parse_args()

    if args.mode == 'test_env':
        print("=== 测试环境 ===")
        test_env()

    elif args.mode == 'test_oracle':
        print("=== 测试Oracle ===")
        test_oracle()

    elif args.mode == 'test_evaluation':
        print("=== 测试评估 ===")
        test_evaluation()

    elif args.mode == 'train':
        print("=== 开始VIPER训练 ===")
        print(f"Oracle路径: {args.oracle_path}")
        print(f"输出路径: {args.output_path}")
        print(f"迭代次数: {args.n_iterations}")
        print(f"每轮采样: {args.samples_per_iter}")
        print(f"树深度: {args.max_depth}")
        print(f"最大叶子: {args.max_leaves}")
        print()

        train_viper(
            oracle_path=args.oracle_path,
            output_path=args.output_path,
            n_iterations=args.n_iterations,
            samples_per_iter=args.samples_per_iter,
            max_depth=args.max_depth,
            max_leaves=args.max_leaves
        )