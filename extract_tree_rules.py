#!/usr/bin/env python3
"""
从训练好的决策树中提取并简化规则

这个脚本展示如何使用规则提取器从VIPER训练的决策树中提取可解释的规则。
规则提取使用统计学方法（卡方检验、Fisher精确检验）来简化规则。

使用示例:
    # 从训练好的决策树提取规则
    python extract_tree_rules.py --tree-path log/viper_TicTacToe-v0_all-leaves_50.joblib

    # 指定输出文件
    python extract_tree_rules.py --tree-path log/viper_TicTacToe-v0_all-leaves_50.joblib \
                                  --output rules_output.txt

    # 使用训练数据进行规则简化
    python extract_tree_rules.py --tree-path log/viper_TicTacToe-v0_all-leaves_50.joblib \
                                  --env-name TicTacToe-v0 \
                                  --oracle-path log/oracle_TicTacToe_selfplay.zip \
                                  --n-samples 5000

作者: VIPER项目组
"""

import argparse
import numpy as np
import sys
from pathlib import Path

from model.tree_wrapper import TreeWrapper

# 导入环境注册（注册TicTacToe-v0等环境）
import gym_env


def collect_training_data(args, n_samples=1000):
    """
    收集训练数据用于规则提取

    Args:
        args: 命令行参数
        n_samples: 采样数量

    Returns:
        (X, y): 特征和标签
    """
    print(f"\n收集 {n_samples} 个样本用于规则分析...")

    # 直接加载Oracle和环境
    from sb3_contrib import MaskablePPO
    import gymnasium as gym

    env = gym.make(args.env_name, opponent_type=args.tictactoe_opponent)
    oracle = MaskablePPO.load(args.oracle_path, env=env)

    X_samples = []
    y_samples = []

    obs, _ = env.reset()
    collected = 0

    while collected < n_samples:
        # 计算mask
        mask = (obs == 0).astype(bool)

        # 使用Oracle预测动作
        import torch
        mask_tensor = torch.tensor(mask).unsqueeze(0)
        action, _ = oracle.predict(obs, deterministic=True, action_masks=mask_tensor)

        X_samples.append(obs.copy())
        y_samples.append(action)
        collected += 1

        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if done:
            obs, _ = env.reset()

    X = np.array(X_samples)
    y = np.array(y_samples)

    print(f"收集完成: X shape={X.shape}, y shape={y.shape}")
    print(f"动作分布: {np.bincount(y)}")

    return X, y


def main():
    parser = argparse.ArgumentParser(
        description="从决策树中提取并简化规则",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础用法（不进行统计简化）
  python extract_tree_rules.py --tree-path log/viper_TicTacToe-v0_all-leaves_50.joblib

  # 使用训练数据进行规则简化
  python extract_tree_rules.py --tree-path log/viper_TicTacToe-v0_all-leaves_50.joblib \\
                                --env-name TicTacToe-v0 \\
                                --oracle-path log/oracle_TicTacToe_selfplay.zip \\
                                --n-samples 5000 \\
                                --alpha 0.05
        """
    )

    # 必需参数
    parser.add_argument("--tree-path", type=str, required=True,
                       help="决策树模型路径 (.joblib)")

    # 可选参数 - 输出
    parser.add_argument("--output", type=str, default=None,
                       help="输出文件路径（默认：与tree-path同名的.txt文件）")
    parser.add_argument("--max-rules", type=int, default=None,
                       help="打印的最大规则数量（None表示全部打印）")

    # 可选参数 - 规则简化
    parser.add_argument("--env-name", type=str, default='TicTacToe-v0',
                       help="环境名称（用于收集训练数据进行规则简化）")
    parser.add_argument("--oracle-path", type=str, default=None,
                       help="Oracle模型路径（用于生成训练数据）")
    parser.add_argument("--n-samples", type=int, default=5000,
                       help="用于规则简化的样本数量（默认5000）")
    parser.add_argument("--alpha", type=float, default=0.05,
                       help="统计检验显著性水平（默认0.05）")

    # 可选参数 - 规则优先级
    parser.add_argument("--compute-priority", action='store_true',
                       help="计算规则优先级（基于状态重要性）")
    parser.add_argument("--sort-by-priority", action='store_true',
                       help="按优先级对规则进行排序（需要--compute-priority）")

    # 环境参数
    parser.add_argument("--n-env", type=int, default=8,
                       help="并行环境数量")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--tictactoe-opponent", type=str, default='random',
                       choices=['random', 'minmax', 'selfplay'],
                       help="井字棋对手类型")

    # 其他参数
    parser.add_argument("--no-simplify", action='store_true',
                       help="不进行规则简化（仅提取）")
    parser.add_argument("--verbose", action='store_true',
                       help="显示详细信息")

    args = parser.parse_args()

    # 检查树文件是否存在
    tree_path = Path(args.tree_path)
    if not tree_path.exists():
        print(f"错误: 树文件不存在: {args.tree_path}")
        sys.exit(1)

    # 设置输出路径
    if args.output is None:
        args.output = tree_path.with_suffix('.rules.txt')

    print("="*80)
    print("决策树规则提取器")
    print("="*80)
    print(f"输入文件: {args.tree_path}")
    print(f"输出文件: {args.output}")
    print(f"显著性水平: {args.alpha}")
    print("="*80)

    # 加载决策树
    print("\n加载决策树...")
    tree_wrapper = TreeWrapper.load(args.tree_path)
    tree_wrapper.print_info()

    # 判断是否需要规则简化或优先级计算
    need_simplification = (not args.no_simplify and
                          args.env_name is not None and
                          args.oracle_path is not None)

    need_priority = args.compute_priority

    # 检查优先级计算的必要参数
    if need_priority and (args.oracle_path is None or args.env_name is None):
        print("错误: 计算优先级需要提供 --oracle-path 和 --env-name 参数")
        sys.exit(1)

    # 加载oracle和环境（仅用于优先级计算，不需要采样）
    oracle = None
    env = None
    if need_priority:
        print("\n加载Oracle模型用于优先级计算...")
        from sb3_contrib import MaskablePPO
        import gymnasium as gym

        env = gym.make(args.env_name, opponent_type=args.tictactoe_opponent)
        oracle = MaskablePPO.load(args.oracle_path, env=env)
        print("✓ Oracle模型加载成功")

    # 准备训练数据（仅规则简化需要）
    X_train = None
    y_train = None
    if need_simplification:
        print("\n收集训练数据用于规则简化...")
        X_train, y_train = collect_training_data(args, n_samples=args.n_samples)
    else:
        # 不需要简化，创建虚拟数据（DecisionTreeRuleExtractor需要X_train来查找规则对应的状态）
        print("\n创建虚拟训练数据（用于规则提取和优先级计算）...")
        X_train = np.zeros((10, tree_wrapper.tree.n_features_in_))
        y_train = np.zeros(10, dtype=int)

    tree_wrapper.set_training_data(X_train, y_train)

    # 提取规则
    print("\n" + "="*80)
    print("开始规则提取")
    print("="*80)

    from model.rule_extractor import DecisionTreeRuleExtractor

    extractor = DecisionTreeRuleExtractor(
        tree_wrapper.tree, X_train, y_train,
        oracle_model=oracle, env=env
    )

    # 1. 提取规则
    extractor.extract_rules(verbose=True)

    # 2. 如果需要，进行简化
    if need_simplification:
        print("\n" + "="*80)
        print("步骤 2: 简化规则")
        print("="*80)
        extractor.simplify_rules(verbose=True)

    # 3. 如果需要，计算优先级
    if need_priority:
        print("\n" + "="*80)
        print("步骤 3: 计算规则优先级（不需要采样）")
        print("="*80)
        extractor.compute_rule_priorities(verbose=True)

        if args.sort_by_priority:
            print("\n按优先级排序规则（降序）...")
            extractor.sort_rules_by_priority(descending=True)

    # 打印统计信息
    stats = extractor.get_stats()
    print("\n" + "="*80)
    print("规则提取统计")
    print("="*80)
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 打印规则
    print("\n" + "="*80)
    print("提取的规则")
    print("="*80)
    extractor.print_rules(max_rules=args.max_rules)

    # 导出规则
    tree_wrapper._rule_extractor = extractor  # 设置提取器

    # 导出为文本文件
    extractor.export_rules_to_text(str(args.output), include_vectors=True)

    # 如果需要，也导出为JSON
    json_output = str(args.output).replace('.txt', '.json')
    if json_output != str(args.output):
        extractor.export_rules_to_json(json_output)

    print("\n" + "="*80)
    print("完成！")
    print("="*80)
    print(f"规则已保存到: {args.output}")
    print(f"总共提取 {len(extractor.rules)} 条规则")

    stats = extractor.get_stats()
    if need_simplification:
        if 'n_removed_antecedents' in stats:
            print(f"简化过程中删除了 {stats['n_removed_antecedents']} 个前件")
        if 'default_consequent' in stats:
            print(f"最常见的动作: {stats['default_consequent']}")

    if need_priority:
        if 'priority_min' in stats and 'priority_max' in stats:
            print(f"优先级范围: [{stats['priority_min']:.4f}, {stats['priority_max']:.4f}]")
            print(f"平均优先级: {stats['priority_mean']:.4f}")
        if args.sort_by_priority:
            print("规则已按优先级降序排序")


if __name__ == "__main__":
    main()
