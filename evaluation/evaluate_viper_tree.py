"""
评估 VIPER 决策树性能

支持：
- 对战不同对手（Random, MinMax）
- 详细统计（胜/负/平/非法移动）
- 可视化决策树规则
- 导出规则到文本文件
"""

import argparse
import numpy as np
import joblib
import gymnasium as gym
from sklearn.tree import export_text
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gym_env


class ProbabilityMaskedTreeWrapper:
    """
    带概率掩码的分类树包装器（用于TicTacToe）
    """

    def __init__(self, tree_model):
        self.tree = tree_model
        self.n_actions = 9  # TicTacToe固定9个动作

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        """预测动作（带合法动作掩码）"""
        # 处理输入shape
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
            single_obs = True
        else:
            single_obs = False

        # 获取概率分布
        action_probs = self.tree.predict_proba(observation)

        # 对每个环境选择最佳合法动作
        actions = []
        for i in range(observation.shape[0]):
            obs = observation[i]
            probs = action_probs[i]

            # 获取合法动作（空位置）
            legal_actions = np.where(obs == 0)[0]

            if len(legal_actions) == 0:
                action = 0
            else:
                # 创建掩码概率
                masked_probs = np.full(self.n_actions, -np.inf)
                masked_probs[legal_actions] = probs[legal_actions]
                action = np.argmax(masked_probs)

            actions.append(action)

        actions = np.array(actions)

        if single_obs:
            return actions[0], None
        else:
            return actions, None

    def print_info(self):
        """打印模型信息"""
        print("\n" + "="*70)
        print("Decision Tree Information")
        print("="*70)
        print(f"Number of leaves: {self.tree.tree_.n_leaves}")
        print(f"Tree depth: {self.tree.tree_.max_depth}")
        print(f"Number of classes: {self.tree.n_classes_}")
        print(f"Classes: {list(self.tree.classes_)}")
        print("="*70 + "\n")


def load_viper_tree(model_path):
    """加载 VIPER 决策树"""
    print(f"加载模型: {model_path}")
    tree = joblib.load(model_path)
    wrapper = ProbabilityMaskedTreeWrapper(tree)
    wrapper.print_info()
    return wrapper


def render_board(board):
    """渲染棋盘（用于调试）"""
    symbols = {-1: 'O', 0: '.', 1: 'X'}
    lines = []
    for i in range(3):
        row = board[i*3:(i+1)*3]
        line = ' '.join([symbols[x] for x in row])
        lines.append(line)
    return '\n'.join(lines)


def evaluate_vs_opponent(policy, opponent_type='minmax', n_episodes=100, verbose=False):
    """
    评估策略对战特定对手

    Args:
        policy: 策略模型
        opponent_type: 对手类型 ('random', 'minmax')
        n_episodes: 评估局数
        verbose: 是否打印详细信息

    Returns:
        stats: 字典，包含 wins, losses, draws, illegal_moves, mean_reward
    """
    env = gym.make('TicTacToe-v0', opponent_type=opponent_type)

    wins, losses, draws, illegal_moves = 0, 0, 0, 0
    total_reward = 0.0

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            if verbose and episode < 3:  # 打印前3局
                print(f"\nEpisode {episode+1}, Step")
                print(render_board(obs))
                print(f"Action: {action}, Reward: {reward}")

            if done:
                total_reward += episode_reward

                # 统计结果
                if 'illegal_move' in info and info['illegal_move']:
                    illegal_moves += 1
                    losses += 1  # 非法移动视为失败
                elif reward > 0:
                    wins += 1
                elif reward < 0:
                    losses += 1
                else:
                    draws += 1

            obs = next_obs

    env.close()

    mean_reward = total_reward / n_episodes

    return {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'illegal_moves': illegal_moves,
        'total': n_episodes,
        'mean_reward': mean_reward,
        'win_rate': wins / n_episodes,
        'draw_rate': draws / n_episodes,
        'loss_rate': losses / n_episodes,
    }


def print_evaluation_results(stats, opponent_type):
    """打印评估结果"""
    print("\n" + "="*70)
    print(f"Evaluation Results vs {opponent_type.upper()}")
    print("="*70)
    print(f"Total games: {stats['total']}")
    print(f"Wins:        {stats['wins']:3d} ({stats['win_rate']*100:5.1f}%)")
    print(f"Draws:       {stats['draws']:3d} ({stats['draw_rate']*100:5.1f}%)")
    print(f"Losses:      {stats['losses']:3d} ({stats['loss_rate']*100:5.1f}%)")
    print(f"Illegal:     {stats['illegal_moves']:3d}")
    print(f"Mean reward: {stats['mean_reward']:6.3f}")
    print("="*70)

    # 性能评价
    if stats['illegal_moves'] > 0:
        print("⚠  有非法移动 - 检查 masking 是否正常工作")
    elif opponent_type == 'minmax':
        if stats['draw_rate'] >= 0.8:
            print("✓  优秀！高平局率说明学到了接近最优策略。")
        elif stats['draw_rate'] >= 0.6:
            print("△  良好，但还有提升空间。")
        else:
            print("✗  需要更多训练或调整参数。")
    elif opponent_type == 'random':
        if stats['win_rate'] >= 0.9:
            print("✓  优秀！对 Random 对手胜率很高。")
        elif stats['win_rate'] >= 0.7:
            print("△  良好，但还有提升空间。")
        else:
            print("✗  需要更多训练或调整参数。")

    print()


def export_tree_rules(tree, output_path):
    """导出决策树规则到文本文件"""
    # 生成特征名称
    feature_names = [f"pos_{i}" for i in range(9)]

    # 导出文本格式
    tree_rules = export_text(
        tree.tree,
        feature_names=feature_names,
        class_names=[str(i) for i in range(9)],
        decimals=2,
        show_weights=True
    )

    # 保存到文件
    with open(output_path, 'w') as f:
        f.write("TicTacToe Decision Tree Rules\n")
        f.write("="*70 + "\n\n")
        f.write("Feature names: pos_0 to pos_8 (board positions)\n")
        f.write("  -1: Opponent's piece (O)\n")
        f.write("   0: Empty\n")
        f.write("   1: Agent's piece (X)\n\n")
        f.write("Class names: 0-8 (action indices)\n\n")
        f.write("="*70 + "\n\n")
        f.write(tree_rules)

    print(f"✓ Tree rules exported to: {output_path}")


def visualize_sample_decisions(policy, n_samples=5):
    """可视化一些样本决策"""
    print("\n" + "="*70)
    print("Sample Decisions")
    print("="*70)

    env = gym.make('TicTacToe-v0', opponent_type='minmax')

    for i in range(n_samples):
        obs, _ = env.reset()
        done = False
        step_count = 0

        print(f"\n--- Game {i+1} ---")

        while not done and step_count < 5:  # 最多显示5步
            print(f"\nStep {step_count + 1}:")
            print(render_board(obs))

            # 获取决策
            action, _ = policy.predict(obs, deterministic=True)

            # 获取概率分布
            probs = policy.tree.predict_proba(obs.reshape(1, -1))[0]
            legal_actions = np.where(obs == 0)[0]

            print(f"Legal actions: {legal_actions}")
            print(f"Action probabilities (legal only):")
            for a in legal_actions:
                print(f"  Action {a}: {probs[a]:.3f}")
            print(f"→ Chosen action: {action}")

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1

    env.close()
    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate VIPER Decision Tree')

    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to VIPER tree model (.joblib)")
    parser.add_argument("--opponent", type=str, default="both",
                       choices=['random', 'minmax', 'both'],
                       help="Opponent type to evaluate against")
    parser.add_argument("--n-episodes", type=int, default=100,
                       help="Number of episodes to evaluate")
    parser.add_argument("--export-rules", type=str, default=None,
                       help="Export tree rules to text file")
    parser.add_argument("--visualize", action="store_true",
                       help="Visualize sample decisions")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed information")

    args = parser.parse_args()

    # 加载模型
    policy = load_viper_tree(args.model_path)

    # 评估对手
    opponents = ['random', 'minmax'] if args.opponent == 'both' else [args.opponent]

    for opponent_type in opponents:
        stats = evaluate_vs_opponent(
            policy, opponent_type,
            n_episodes=args.n_episodes,
            verbose=args.verbose
        )
        print_evaluation_results(stats, opponent_type)

    # 导出规则
    if args.export_rules:
        export_tree_rules(policy.tree, args.export_rules)

    # 可视化决策
    if args.visualize:
        visualize_sample_decisions(policy, n_samples=3)


if __name__ == "__main__":
    main()
