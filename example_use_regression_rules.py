#!/usr/bin/env python3
"""
示例：使用回归树规则进行决策

演示如何加载和使用包含9维向量的回归树规则，
并自动排除不合法的落子。
"""

import json
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_rules(json_path):
    """加载规则JSON文件"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f"加载规则: {json_path}")
    print(f"  树类型: {data['tree_type']}")
    print(f"  规则数量: {len(data['rules'])}")
    if 'statistics' in data:
        print(f"  优先级范围: [{data['statistics'].get('priority_min', 0):.4f}, "
              f"{data['statistics'].get('priority_max', 0):.4f}]")
    print()
    return data['rules']


def find_matching_rule(observation, rules):
    """
    找到匹配当前状态的规则

    Args:
        observation: 状态向量 (9维)
        rules: 规则列表

    Returns:
        匹配的规则，如果没有匹配则返回None
    """
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


def select_action(observation, rules, verbose=False):
    """
    根据规则选择动作，自动排除不合法落子

    Args:
        observation: 当前棋盘状态 (9维向量)
        rules: 规则列表
        verbose: 是否打印详细信息

    Returns:
        最佳合法动作，如果没有找到规则则返回None
    """
    # 1. 找到匹配的规则
    rule = find_matching_rule(observation, rules)

    if rule is None:
        if verbose:
            print("警告: 没有找到匹配的规则")
        return None

    # 2. 获取输出向量
    output_vector = np.array(rule['output_vector'])

    if verbose:
        print(f"匹配规则:")
        print(f"  条件数量: {len(rule['antecedents'])}")
        print(f"  输出向量: {output_vector}")
        if 'priority' in rule:
            print(f"  优先级: {rule['priority']:.4f}")

    # 3. 计算合法动作mask（井字棋：空位置为合法）
    mask = (observation == 0)

    if verbose:
        print(f"  合法动作: {np.where(mask)[0].tolist()}")

    # 4. 应用mask：将不合法动作设为-inf
    logits = output_vector.copy()
    logits[~mask] = -np.inf

    # 5. 选择最佳合法动作
    best_action = np.argmax(logits)

    if verbose:
        print(f"  选择动作: {best_action}")
        print(f"  该动作的logit: {output_vector[best_action]:.4f}")

    return best_action


def visualize_board(observation):
    """可视化井字棋棋盘"""
    symbols = {0: '.', 1: 'X', -1: 'O'}
    board = observation.reshape(3, 3)
    print("\n当前棋盘:")
    print("  0 1 2")
    for i in range(3):
        row = ' '.join([symbols[board[i, j]] for j in range(3)])
        print(f"{i} {row}")
    print()


def demo_basic():
    """基础演示"""
    print("="*80)
    print("示例 1: 基础用法")
    print("="*80)

    # 示例棋盘状态：
    # . X .
    # . O .
    # . . X
    observation = np.array([0, 1, 0, 0, -1, 0, 0, 0, 1])

    # 创建示例规则（在实际使用中应该从JSON加载）
    rules = [
        {
            'antecedents': [[4, '<=', 0.5], [1, '>', 0.5]],
            'output_vector': [0.8, -0.2, 0.5, 0.3, -0.9, 0.6, 0.4, 0.7, -0.1],
            'best_action': 0,
            'priority': 1.5
        }
    ]

    visualize_board(observation)

    print("使用规则进行决策...")
    action = select_action(observation, rules, verbose=True)

    if action is not None:
        print(f"\n✓ 选择动作: {action}")
        print(f"  对应位置: ({action // 3}, {action % 3})")
    else:
        print("\n✗ 未找到匹配规则")


def demo_from_file(json_path):
    """从JSON文件加载规则并演示"""
    print("\n" + "="*80)
    print("示例 2: 从JSON文件加载规则")
    print("="*80)

    # 检查文件是否存在
    if not os.path.exists(json_path):
        print(f"错误: 文件不存在: {json_path}")
        print("提示: 请先运行 extract_tree_rules.py 生成规则文件")
        return

    # 加载规则
    rules = load_rules(json_path)

    # 测试多个棋盘状态
    test_cases = [
        # 案例1: 开局
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),

        # 案例2: 中局
        np.array([0, 1, 0, 0, -1, 0, 0, 0, 1]),

        # 案例3: 关键一步
        np.array([1, 1, 0, -1, -1, 0, 0, 0, 1]),
    ]

    for i, observation in enumerate(test_cases, 1):
        print(f"\n{'─'*80}")
        print(f"测试案例 {i}:")
        visualize_board(observation)

        action = select_action(observation, rules, verbose=True)

        if action is not None:
            print(f"\n✓ 选择落子位置: ({action // 3}, {action % 3})")
        else:
            print("\n✗ 未找到匹配规则")


def demo_compare_with_without_mask():
    """演示mask的作用"""
    print("\n" + "="*80)
    print("示例 3: 对比有无mask的区别")
    print("="*80)

    # 棋盘状态：中间和几个角已占
    # X . .
    # . O .
    # . . X
    observation = np.array([1, 0, 0, 0, -1, 0, 0, 0, 1])

    output_vector = np.array([0.1, 0.9, 0.3, 0.7, 0.2, 0.5, 0.8, 0.6, 0.4])

    visualize_board(observation)

    print("输出向量:", output_vector)
    print()

    # 不考虑mask
    print("1. 不考虑合法性（错误做法）:")
    best_action_no_mask = np.argmax(output_vector)
    print(f"   选择动作: {best_action_no_mask}")
    print(f"   问题: 位置 {best_action_no_mask} 已被占据！")
    print()

    # 考虑mask
    print("2. 考虑合法性（正确做法）:")
    mask = (observation == 0)
    logits = output_vector.copy()
    logits[~mask] = -np.inf
    best_action_with_mask = np.argmax(logits)
    print(f"   合法位置: {np.where(mask)[0].tolist()}")
    print(f"   选择动作: {best_action_with_mask}")
    print(f"   ✓ 位置 {best_action_with_mask} 是空的，合法！")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description='演示如何使用回归树规则进行决策'
    )
    parser.add_argument(
        '--rules',
        type=str,
        default=None,
        help='规则JSON文件路径（可选）'
    )
    parser.add_argument(
        '--demo',
        type=str,
        default='all',
        choices=['basic', 'file', 'mask', 'all'],
        help='运行哪个演示'
    )

    args = parser.parse_args()

    print("\n回归树规则使用演示")
    print("="*80)

    if args.demo in ['basic', 'all']:
        demo_basic()

    if args.demo in ['mask', 'all']:
        demo_compare_with_without_mask()

    if args.demo in ['file', 'all'] and args.rules:
        demo_from_file(args.rules)
    elif args.demo == 'file' and not args.rules:
        print("\n提示: 使用 --rules 参数指定规则JSON文件")
        print("示例: python example_use_regression_rules.py --rules rules.json")

    print("\n" + "="*80)
    print("演示完成")
    print("="*80)


if __name__ == "__main__":
    main()
