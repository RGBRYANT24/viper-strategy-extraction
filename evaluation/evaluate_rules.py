#!/usr/bin/env python3
"""
评估规则策略的性能

这个脚本加载从决策树提取的规则（JSON格式），
使用规则策略与不同对手对战，评估胜率。

使用示例:
    # 对战random对手
    python evaluate_rules.py --rules rules.json --opponent random --n-episodes 100

    # 对战minmax对手
    python evaluate_rules.py --rules rules.json --opponent minmax --n-episodes 100

    # 完整评估（对战所有对手类型）
    python evaluate_rules.py --rules rules.json --eval-all --n-episodes 100

    # 与决策树对比
    python evaluate_rules.py --rules rules.json --tree-path log/xxx.joblib --opponent random
"""

import argparse
import json
import numpy as np
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# 导入环境注册
import gym_env
import gymnasium as gym


class RuleBasedPolicy:
    """基于规则的策略"""

    def __init__(self, rules_path: str, top_k: Optional[int] = None):
        """
        初始化规则策略

        Args:
            rules_path: 规则JSON文件路径
            top_k: 只使用优先级最高的top_k条规则（None表示使用所有规则）
        """
        with open(rules_path, 'r') as f:
            data = json.load(f)

        self.rules = data['rules']
        self.tree_type = data.get('tree_type', 'regressor')

        # 如果指定了top_k，只保留前k条规则（假设已经按优先级排序）
        if top_k is not None and top_k < len(self.rules):
            self.rules = self.rules[:top_k]
            print(f"使用优先级最高的前 {top_k} 条规则")

        print(f"加载了 {len(self.rules)} 条规则")
        if self.rules:
            priorities = [r.get('priority', 0.0) for r in self.rules]
            print(f"优先级范围: [{min(priorities):.4f}, {max(priorities):.4f}]")

    def find_matching_rule(self, observation: np.ndarray) -> Optional[Dict]:
        """找到匹配当前状态的规则"""
        for rule in self.rules:
            # 检查所有前件条件
            match = True
            for feature_idx, operator, value in rule['antecedents']:
                if operator == '<=':
                    if not (observation[feature_idx] <= value):
                        match = False
                        break
                else:  # operator == '>'
                    if not (observation[feature_idx] > value):
                        match = False
                        break

            if match:
                return rule

        return None

    def predict(self, observation: np.ndarray, mask: np.ndarray = None) -> int:
        """
        预测动作

        Args:
            observation: 当前状态
            mask: 合法动作mask（True表示合法）

        Returns:
            选择的动作
        """
        # 找到匹配的规则
        rule = self.find_matching_rule(observation)

        if rule is None:
            # 没有匹配规则，随机选择合法动作
            raise RuntimeError("没有匹配的规则！")
            # if mask is not None:
            #     legal_actions = np.where(mask)[0]
            #     if len(legal_actions) > 0:
            #         return np.random.choice(legal_actions)
            # return np.random.randint(0, 9)

        # 获取输出向量
        if 'output_vector' in rule:
            # 回归树：使用完整的输出向量
            output_vector = np.array(rule['output_vector'])

            # 应用mask
            if mask is not None:
                logits = output_vector.copy()
                logits[~mask] = -np.inf
                return int(np.argmax(logits))
            else:
                return int(np.argmax(output_vector))
        else:
            # 分类树：直接返回consequent
            action = rule.get('best_action', rule.get('consequent', 0))

            # 检查是否合法
            if mask is not None and not mask[action]:
                # 不合法，选择其他合法动作
                legal_actions = np.where(mask)[0]
                if len(legal_actions) > 0:
                    return np.random.choice(legal_actions)

            return int(action)


def evaluate_policy(policy, opponent_type: str, n_episodes: int = 100,
                   verbose: bool = False) -> Dict[str, float]:
    """
    评估策略性能

    Args:
        policy: 策略对象（RuleBasedPolicy或TreeWrapper）
        opponent_type: 对手类型（'random', 'minmax'）
        n_episodes: 对战局数
        verbose: 是否显示详细信息

    Returns:
        包含胜率等统计信息的字典
    """
    env = gym.make('TicTacToe-v0', opponent_type=opponent_type)

    wins = 0
    losses = 0
    draws = 0
    illegal_moves = 0
    total_steps = 0

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_steps = 0

        while not done:
            # 计算合法动作mask
            mask = (obs == 0).astype(bool)

            # 策略预测
            if isinstance(policy, RuleBasedPolicy):
                action = policy.predict(obs, mask)
            else:
                # TreeWrapper
                action = policy.predict(obs)

            # 检查是否合法
            if not mask[action]:
                illegal_moves += 1
                if verbose:
                    print(f"Episode {episode}: 非法动作 {action}")
                # 选择随机合法动作
                legal_actions = np.where(mask)[0]
                if len(legal_actions) > 0:
                    action = np.random.choice(legal_actions)
                else:
                    break

            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_steps += 1

            if done:
                total_steps += episode_steps
                if reward > 0:
                    wins += 1
                elif reward < 0:
                    losses += 1
                else:
                    draws += 1

                if verbose and episode % 10 == 0:
                    print(f"Episode {episode}: reward={reward}, steps={episode_steps}")

    win_rate = wins / n_episodes
    loss_rate = losses / n_episodes
    draw_rate = draws / n_episodes
    avg_steps = total_steps / n_episodes
    illegal_rate = illegal_moves / n_episodes

    return {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'win_rate': win_rate,
        'loss_rate': loss_rate,
        'draw_rate': draw_rate,
        'avg_steps': avg_steps,
        'illegal_moves': illegal_moves,
        'illegal_rate': illegal_rate
    }


def compare_with_tree(rules_policy, tree_path: str, opponent_type: str,
                     n_episodes: int = 100) -> None:
    """
    对比规则策略和决策树策略

    Args:
        rules_policy: 规则策略
        tree_path: 决策树路径
        opponent_type: 对手类型
        n_episodes: 对战局数
    """
    print("\n" + "="*80)
    print(f"规则策略 vs 决策树策略 (对手: {opponent_type})")
    print("="*80)

    # 评估规则策略
    print("\n评估规则策略...")
    rules_stats = evaluate_policy(rules_policy, opponent_type, n_episodes)

    # 评估决策树策略
    print("\n评估决策树策略...")
    from model.tree_wrapper import TreeWrapper
    tree_wrapper = TreeWrapper.load(tree_path)
    tree_stats = evaluate_policy(tree_wrapper, opponent_type, n_episodes)

    # 打印对比结果
    print("\n" + "="*80)
    print("对比结果")
    print("="*80)
    print(f"{'指标':<20} {'规则策略':<15} {'决策树策略':<15} {'差异':<15}")
    print("-"*80)

    metrics = [
        ('胜率', 'win_rate'),
        ('平局率', 'draw_rate'),
        ('败率', 'loss_rate'),
        ('平均步数', 'avg_steps'),
        ('非法动作率', 'illegal_rate')
    ]

    for name, key in metrics:
        rules_val = rules_stats[key]
        tree_val = tree_stats[key]
        diff = rules_val - tree_val

        if key == 'avg_steps':
            print(f"{name:<20} {rules_val:<15.2f} {tree_val:<15.2f} {diff:+.2f}")
        else:
            print(f"{name:<20} {rules_val:<15.2%} {tree_val:<15.2%} {diff:+.2%}")


def main():
    parser = argparse.ArgumentParser(
        description="评估规则策略的性能",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 对战random对手
  python evaluate_rules.py --rules rules.json --opponent random --n-episodes 100

  # 对战minmax对手
  python evaluate_rules.py --rules rules.json --opponent minmax --n-episodes 100

  # 完整评估（对战所有对手类型）
  python evaluate_rules.py --rules rules.json --eval-all --n-episodes 100

  # 与决策树对比
  python evaluate_rules.py --rules rules.json --tree-path log/xxx.joblib --opponent random

  # 只使用优先级最高的前20条规则
  python evaluate_rules.py --rules rules.json --top-k 20 --eval-all
        """
    )

    # 必需参数
    parser.add_argument("--rules", type=str, required=True,
                       help="规则JSON文件路径")

    # 评估参数
    parser.add_argument("--opponent", type=str, default='random',
                       choices=['random', 'minmax'],
                       help="对手类型")
    parser.add_argument("--eval-all", action='store_true',
                       help="评估所有对手类型")
    parser.add_argument("--n-episodes", type=int, default=100,
                       help="对战局数（默认100）")

    # 规则选择
    parser.add_argument("--top-k", type=int, default=None,
                       help="只使用优先级最高的前k条规则（None表示使用所有规则）")

    # 对比参数
    parser.add_argument("--tree-path", type=str, default=None,
                       help="决策树模型路径（用于对比）")

    # 其他参数
    parser.add_argument("--verbose", action='store_true',
                       help="显示详细信息")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")

    args = parser.parse_args()

    # 检查规则文件是否存在
    rules_path = Path(args.rules)
    if not rules_path.exists():
        print(f"错误: 规则文件不存在: {args.rules}")
        sys.exit(1)

    # 设置随机种子
    np.random.seed(args.seed)

    print("="*80)
    print("规则策略评估器")
    print("="*80)
    print(f"规则文件: {args.rules}")
    print(f"对战局数: {args.n_episodes}")
    print("="*80)

    # 加载规则策略
    print("\n加载规则策略...")
    policy = RuleBasedPolicy(args.rules, top_k=args.top_k)

    # 决定要评估的对手类型
    if args.eval_all:
        opponents = ['random', 'minmax']
    else:
        opponents = [args.opponent]

    # 评估每个对手
    all_results = {}
    for opponent in opponents:
        print("\n" + "="*80)
        print(f"评估对手: {opponent}")
        print("="*80)

        stats = evaluate_policy(policy, opponent, args.n_episodes, args.verbose)
        all_results[opponent] = stats

        # 打印结果
        print("\n结果:")
        print(f"  胜: {stats['wins']} ({stats['win_rate']:.1%})")
        print(f"  平: {stats['draws']} ({stats['draw_rate']:.1%})")
        print(f"  负: {stats['losses']} ({stats['loss_rate']:.1%})")
        print(f"  平均步数: {stats['avg_steps']:.2f}")
        print(f"  非法动作: {stats['illegal_moves']} ({stats['illegal_rate']:.2%})")

    # 如果需要与决策树对比
    if args.tree_path is not None:
        if not Path(args.tree_path).exists():
            print(f"\n警告: 决策树文件不存在: {args.tree_path}")
        else:
            for opponent in opponents:
                compare_with_tree(policy, args.tree_path, opponent, args.n_episodes)

    # 总结
    print("\n" + "="*80)
    print("评估完成！")
    print("="*80)

    if len(opponents) > 1:
        print("\n所有对手类型的总结:")
        print(f"{'对手':<15} {'胜率':<10} {'平局率':<10} {'败率':<10}")
        print("-"*50)
        for opponent in opponents:
            stats = all_results[opponent]
            print(f"{opponent:<15} {stats['win_rate']:<10.1%} "
                  f"{stats['draw_rate']:<10.1%} {stats['loss_rate']:<10.1%}")


if __name__ == "__main__":
    main()
