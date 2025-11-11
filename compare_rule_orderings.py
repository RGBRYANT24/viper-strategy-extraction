#!/usr/bin/env python3
"""
对比不同规则排序策略的性能

这个脚本生成不同排序版本的规则集合，评估它们的性能，
验证优先级排序是否真的有效。

使用示例:
    # 对比所有排序策略
    python compare_rule_orderings.py \
        --rules rules.json \
        --n-episodes 100 \
        --opponent random

    # 生成多个随机排序版本
    python compare_rule_orderings.py \
        --rules rules.json \
        --n-random 5 \
        --n-episodes 100 \
        --eval-all
"""

import argparse
import json
import numpy as np
import sys
import random
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

# 导入环境注册
import gym_env
import gymnasium as gym


def load_rules(rules_path: str) -> Dict:
    """加载规则JSON文件"""
    with open(rules_path, 'r') as f:
        data = json.load(f)
    return data


def save_rules(data: Dict, output_path: str):
    """保存规则JSON文件"""
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"规则已保存到: {output_path}")


def create_random_ordering(rules: List[Dict], seed: int = None) -> List[Dict]:
    """创建随机排序的规则"""
    if seed is not None:
        random.seed(seed)
    shuffled = rules.copy()
    random.shuffle(shuffled)
    return shuffled


def create_reverse_priority_ordering(rules: List[Dict]) -> List[Dict]:
    """创建按优先级逆序排列的规则（最低优先级在前）"""
    return sorted(rules, key=lambda r: r.get('priority', 0.0), reverse=False)


def create_support_ordering(rules: List[Dict]) -> List[Dict]:
    """创建按支持度排序的规则（支持度高的在前）"""
    return sorted(rules, key=lambda r: r.get('support_count', 0), reverse=True)


def find_matching_rule(observation: np.ndarray, rules: List[Dict]) -> Dict:
    """找到第一个匹配当前状态的规则"""
    for rule in rules:
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


def evaluate_rules(rules: List[Dict], opponent_type: str, n_episodes: int = 100) -> Dict:
    """评估规则策略"""
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

            # 找到匹配的规则
            rule = find_matching_rule(obs, rules)

            if rule is None:
                # 没有匹配规则，随机选择
                legal_actions = np.where(mask)[0]
                if len(legal_actions) > 0:
                    action = np.random.choice(legal_actions)
                else:
                    break
            else:
                # 使用规则
                output_vector = np.array(rule['output_vector'])
                logits = output_vector.copy()
                logits[~mask] = -np.inf
                action = int(np.argmax(logits))

            # 检查合法性
            if not mask[action]:
                illegal_moves += 1
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

    return {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'win_rate': wins / n_episodes,
        'loss_rate': losses / n_episodes,
        'draw_rate': draws / n_episodes,
        'avg_steps': total_steps / n_episodes,
        'illegal_moves': illegal_moves
    }


def main():
    parser = argparse.ArgumentParser(
        description="对比不同规则排序策略的性能",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # 必需参数
    parser.add_argument("--rules", type=str, required=True,
                       help="规则JSON文件路径（应该是按优先级排序的）")

    # 评估参数
    parser.add_argument("--opponent", type=str, default='random',
                       choices=['random', 'minmax'],
                       help="对手类型")
    parser.add_argument("--eval-all", action='store_true',
                       help="评估所有对手类型")
    parser.add_argument("--n-episodes", type=int, default=100,
                       help="每个策略的对战局数")

    # 随机排序参数
    parser.add_argument("--n-random", type=int, default=3,
                       help="生成多少个随机排序版本")

    # 输出参数
    parser.add_argument("--save-orderings", action='store_true',
                       help="保存不同排序的规则文件")
    parser.add_argument("--output-dir", type=str, default="rule_orderings",
                       help="保存排序规则的目录")

    # 其他参数
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")

    args = parser.parse_args()

    # 检查规则文件
    rules_path = Path(args.rules)
    if not rules_path.exists():
        print(f"错误: 规则文件不存在: {args.rules}")
        sys.exit(1)

    # 设置随机种子
    np.random.seed(args.seed)
    random.seed(args.seed)

    print("="*80)
    print("规则排序策略对比")
    print("="*80)
    print(f"规则文件: {args.rules}")
    print(f"对战局数: {args.n_episodes}")
    print(f"随机排序版本: {args.n_random}")
    print("="*80)

    # 加载规则
    print("\n加载规则...")
    data = load_rules(args.rules)
    original_rules = data['rules']
    print(f"加载了 {len(original_rules)} 条规则")

    # 检查规则是否有优先级
    priorities = [r.get('priority', 0.0) for r in original_rules]
    print(f"优先级范围: [{min(priorities):.4f}, {max(priorities):.4f}]")

    # 准备不同排序策略
    orderings = {
        'priority_desc': {
            'name': '优先级降序（原始）',
            'rules': original_rules,
            'description': '按状态重要性降序排列'
        },
        'priority_asc': {
            'name': '优先级升序（逆序）',
            'rules': create_reverse_priority_ordering(original_rules),
            'description': '最低优先级在前'
        },
        'support': {
            'name': '支持度排序',
            'rules': create_support_ordering(original_rules),
            'description': '按训练样本支持度排序'
        }
    }

    # 生成多个随机排序
    for i in range(args.n_random):
        orderings[f'random_{i+1}'] = {
            'name': f'随机排序 #{i+1}',
            'rules': create_random_ordering(original_rules, seed=args.seed + i),
            'description': f'随机种子: {args.seed + i}'
        }

    # 如果需要，保存排序后的规则
    if args.save_orderings:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        print(f"\n保存排序后的规则到: {output_dir}")

        for key, ordering in orderings.items():
            output_data = data.copy()
            output_data['rules'] = ordering['rules']
            output_data['ordering_type'] = ordering['name']
            output_data['ordering_description'] = ordering['description']

            output_path = output_dir / f"rules_{key}.json"
            save_rules(output_data, str(output_path))

    # 决定要评估的对手
    if args.eval_all:
        opponents = ['random', 'minmax']
    else:
        opponents = [args.opponent]

    # 评估每种排序策略
    print("\n" + "="*80)
    print("开始评估")
    print("="*80)

    all_results = defaultdict(dict)

    for opponent in opponents:
        print(f"\n{'='*80}")
        print(f"对手: {opponent}")
        print(f"{'='*80}")

        for key, ordering in orderings.items():
            print(f"\n评估: {ordering['name']}")
            print(f"  {ordering['description']}")

            results = evaluate_rules(ordering['rules'], opponent, args.n_episodes)
            all_results[opponent][key] = {
                'name': ordering['name'],
                'results': results
            }

            print(f"  胜率: {results['win_rate']:.1%}")
            print(f"  平局率: {results['draw_rate']:.1%}")
            print(f"  败率: {results['loss_rate']:.1%}")

    # 统计分析
    print("\n" + "="*80)
    print("结果总结")
    print("="*80)

    for opponent in opponents:
        print(f"\n对手: {opponent}")
        print("-"*80)
        print(f"{'排序策略':<30} {'胜率':<10} {'平局率':<10} {'败率':<10}")
        print("-"*80)

        # 按胜率排序
        sorted_results = sorted(
            all_results[opponent].items(),
            key=lambda x: x[1]['results']['win_rate'],
            reverse=True
        )

        for key, data in sorted_results:
            name = data['name']
            r = data['results']
            print(f"{name:<30} {r['win_rate']:<10.1%} {r['draw_rate']:<10.1%} {r['loss_rate']:<10.1%}")

    # 统计显著性分析
    print("\n" + "="*80)
    print("统计分析")
    print("="*80)

    for opponent in opponents:
        print(f"\n对手: {opponent}")

        # 计算优先级排序 vs 随机排序的平均差异
        priority_winrate = all_results[opponent]['priority_desc']['results']['win_rate']

        random_winrates = []
        for i in range(args.n_random):
            random_winrates.append(
                all_results[opponent][f'random_{i+1}']['results']['win_rate']
            )

        avg_random_winrate = np.mean(random_winrates)
        std_random_winrate = np.std(random_winrates)

        print(f"  优先级排序胜率: {priority_winrate:.1%}")
        print(f"  随机排序平均胜率: {avg_random_winrate:.1%} ± {std_random_winrate:.1%}")
        print(f"  差异: {(priority_winrate - avg_random_winrate):.1%}")

        if priority_winrate > avg_random_winrate:
            print(f"  ✓ 优先级排序优于随机排序")
        else:
            print(f"  ✗ 优先级排序未优于随机排序")

        # 对比逆序
        reverse_winrate = all_results[opponent]['priority_asc']['results']['win_rate']
        print(f"\n  优先级逆序胜率: {reverse_winrate:.1%}")
        print(f"  差异: {(priority_winrate - reverse_winrate):.1%}")

        if priority_winrate > reverse_winrate:
            print(f"  ✓ 优先级排序优于逆序排序")
        else:
            print(f"  ✗ 优先级排序未优于逆序排序")

    print("\n" + "="*80)
    print("完成！")
    print("="*80)

    # 结论
    print("\n结论:")
    for opponent in opponents:
        priority_winrate = all_results[opponent]['priority_desc']['results']['win_rate']
        avg_random_winrate = np.mean([
            all_results[opponent][f'random_{i+1}']['results']['win_rate']
            for i in range(args.n_random)
        ])

        improvement = (priority_winrate - avg_random_winrate) * 100

        print(f"  对手={opponent}: 优先级排序比随机排序平均提升 {improvement:+.1f} 个百分点")

        if improvement > 0:
            print(f"    → 优先级函数有效！")
        else:
            print(f"    → 优先级函数可能需要改进")


if __name__ == "__main__":
    main()
