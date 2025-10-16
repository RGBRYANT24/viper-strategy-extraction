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
from gym_env import make_env
from train.viper import load_oracle_env


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

    env, oracle = load_oracle_env(args)

    X_samples = []
    y_samples = []

    obs = env.reset()
    collected = 0

    while collected < n_samples:
        # 使用Oracle预测动作
        action, _ = oracle.predict(obs, deterministic=True)

        # 处理向量化环境
        if len(obs.shape) > 1 and obs.shape[0] > 1:
            # 向量化环境
            n_envs = obs.shape[0]
            for i in range(n_envs):
                if collected < n_samples:
                    X_samples.append(obs[i])
                    y_samples.append(action[i] if hasattr(action, '__len__') else action)
                    collected += 1
        else:
            # 单个环境
            X_samples.append(obs)
            y_samples.append(action if not hasattr(action, '__len__') else action[0])
            collected += 1

        # 执行动作
        step_result = env.step(action)
        if len(step_result) == 5:
            next_obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            next_obs, reward, done, info = step_result

        obs = next_obs

        # 处理环境结束
        if len(obs.shape) > 1 and obs.shape[0] > 1:
            # 向量化环境
            if hasattr(done, '__len__') and np.any(done):
                obs = env.reset()
        else:
            # 单个环境
            if done:
                obs = env.reset()

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
    parser.add_argument("--env-name", type=str, default=None,
                       help="环境名称（用于收集训练数据进行规则简化）")
    parser.add_argument("--oracle-path", type=str, default=None,
                       help="Oracle模型路径（用于生成训练数据）")
    parser.add_argument("--n-samples", type=int, default=5000,
                       help="用于规则简化的样本数量（默认5000）")
    parser.add_argument("--alpha", type=float, default=0.05,
                       help="统计检验显著性水平（默认0.05）")

    # 环境参数
    parser.add_argument("--n-env", type=int, default=8,
                       help="并行环境数量")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--tictactoe-opponent", type=str, default="selfplay",
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

    # 判断是否需要规则简化
    need_simplification = (not args.no_simplify and
                          args.env_name is not None and
                          args.oracle_path is not None)

    if need_simplification:
        print("\n将使用训练数据进行规则简化")

        # 收集训练数据
        X_train, y_train = collect_training_data(args, n_samples=args.n_samples)

        # 设置训练数据
        tree_wrapper.set_training_data(X_train, y_train)

        # 提取并简化规则
        print("\n" + "="*80)
        print("开始规则提取和简化")
        print("="*80)

        extractor = tree_wrapper.extract_rules(alpha=args.alpha, verbose=True)

        # 打印统计信息
        stats = extractor.get_stats()
        print("\n" + "="*80)
        print("规则提取统计")
        print("="*80)
        for key, value in stats.items():
            print(f"  {key}: {value}")

    else:
        print("\n注意: 未提供训练数据，将只提取规则，不进行统计简化")
        print("提示: 使用 --env-name 和 --oracle-path 参数可以进行规则简化")

        # 只提取规则，不简化
        from model.rule_extractor import DecisionTreeRuleExtractor

        # 创建一个虚拟的训练集（只用于提取规则结构）
        X_dummy = np.zeros((10, tree_wrapper.tree.n_features_in_))
        y_dummy = np.zeros(10, dtype=int)

        extractor = DecisionTreeRuleExtractor(tree_wrapper.tree, X_dummy, y_dummy)
        extractor.extract_rules(verbose=True)

    # 打印规则
    print("\n" + "="*80)
    print("提取的规则")
    print("="*80)
    extractor.print_rules(max_rules=args.max_rules)

    # 导出规则
    tree_wrapper._rule_extractor = extractor  # 设置提取器
    tree_wrapper.export_rules(args.output)

    print("\n" + "="*80)
    print("完成！")
    print("="*80)
    print(f"规则已保存到: {args.output}")
    print(f"总共提取 {len(extractor.rules)} 条规则")

    if need_simplification:
        stats = extractor.get_stats()
        if 'n_removed_antecedents' in stats:
            print(f"简化过程中删除了 {stats['n_removed_antecedents']} 个前件")
        if 'default_consequent' in stats:
            print(f"最常见的动作: {stats['default_consequent']}")


if __name__ == "__main__":
    main()
