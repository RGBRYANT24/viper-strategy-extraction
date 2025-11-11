#!/usr/bin/env python
"""
分析和对比VIPER并行训练的结果
读取训练日志和模型，生成性能对比报告
"""
import os
import sys
import json
import glob
import re
import argparse
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train.viper_mask_ppo import RegressionTreePolicy, evaluate_policy


def parse_model_filename(filename):
    """从文件名解析模型参数

    示例: viper_mask_ppo_tree_20251107_003010_iter10_samples50000_depth15_leaves100.joblib

    Returns:
        dict: 包含 timestamp, iterations, samples, depth, leaves
    """
    basename = os.path.basename(filename)
    parts = basename.replace('.joblib', '').split('_')

    info = {
        'filepath': filename,
        'basename': basename,
    }

    for part in parts:
        if part.startswith('iter'):
            info['iterations'] = int(part.replace('iter', ''))
        elif part.startswith('samples'):
            info['samples'] = int(part.replace('samples', ''))
        elif part.startswith('depth'):
            info['depth'] = int(part.replace('depth', ''))
        elif part.startswith('leaves'):
            info['leaves'] = int(part.replace('leaves', ''))
        elif len(part) >= 14 and part[:8].isdigit():  # timestamp
            info['timestamp'] = part

    return info


def parse_training_log(log_file):
    """解析训练日志文件

    Returns:
        dict: 训练过程信息
    """
    if not os.path.exists(log_file):
        return None

    with open(log_file, 'r') as f:
        content = f.read()

    # 提取配置信息
    info = {}

    # 提取最终最佳结果
    match = re.search(r'Best tree: Iteration (\d+)', content)
    if match:
        info['best_iteration'] = int(match.group(1))

    match = re.search(r'Best reward: ([\d.]+)', content)
    if match:
        info['best_reward'] = float(match.group(1))

    # 提取所有迭代的奖励
    iterations = re.findall(r'Iteration (\d+)/\d+\s+Mean reward: ([\d.]+)', content)
    if iterations:
        info['all_iterations'] = [(int(it), float(reward)) for it, reward in iterations]

    return info


def load_and_evaluate_model(model_path, opponents=['random', 'minmax'], n_episodes=1000):
    """加载模型并评估性能

    Args:
        model_path: 模型文件路径
        opponents: 对手列表
        n_episodes: 评估局数

    Returns:
        dict: 评估结果
    """
    try:
        # 加载模型
        tree = joblib.load(model_path)
        policy = RegressionTreePolicy(tree)

        # 模型信息
        results = {
            'tree_depth': tree.tree_.max_depth,
            'tree_leaves': tree.tree_.n_leaves,
            'n_nodes': tree.tree_.node_count,
        }

        # 评估对战不同对手
        for opponent in opponents:
            eval_result = evaluate_policy(
                policy,
                env_name='TicTacToe-v0',
                opponent_type=opponent,
                n_episodes=n_episodes
            )
            results[f'{opponent}_mean_reward'] = eval_result['mean_reward']
            results[f'{opponent}_std_reward'] = eval_result['std_reward']
            results[f'{opponent}_win_rate'] = eval_result['win_rate']
            results[f'{opponent}_draw_rate'] = eval_result['draw_rate']
            results[f'{opponent}_loss_rate'] = eval_result['loss_rate']
            results[f'{opponent}_wins'] = eval_result['wins']
            results[f'{opponent}_draws'] = eval_result['draws']
            results[f'{opponent}_losses'] = eval_result['losses']

        return results

    except Exception as e:
        print(f"✗ 加载/评估失败 {model_path}: {e}")
        return None


def collect_all_results(models_dir, opponents=['random', 'minmax'], n_episodes=1000,
                       pattern="viper_mask_ppo_tree_*.joblib"):
    """收集目录下所有模型的评估结果

    Args:
        models_dir: 模型目录
        opponents: 对手列表
        n_episodes: 评估局数
        pattern: 模型文件名模式

    Returns:
        pandas.DataFrame: 结果数据框
    """
    # 查找所有模型文件
    model_files = glob.glob(os.path.join(models_dir, pattern))

    if not model_files:
        print(f"未找到模型文件: {os.path.join(models_dir, pattern)}")
        return None

    print(f"找到 {len(model_files)} 个模型文件")

    all_results = []

    for i, model_path in enumerate(model_files, 1):
        print(f"\n[{i}/{len(model_files)}] 评估: {os.path.basename(model_path)}")

        # 解析文件名
        model_info = parse_model_filename(model_path)

        # 加载并评估
        eval_results = load_and_evaluate_model(model_path, opponents, n_episodes)

        if eval_results:
            # 合并信息
            result = {**model_info, **eval_results}
            all_results.append(result)

            # 打印简要结果
            if 'random' in opponents:
                print(f"  vs Random: Win {eval_results['random_win_rate']*100:.1f}% | "
                      f"Draw {eval_results['random_draw_rate']*100:.1f}%")
            if 'minmax' in opponents:
                print(f"  vs MinMax: Win {eval_results['minmax_win_rate']*100:.1f}% | "
                      f"Draw {eval_results['minmax_draw_rate']*100:.1f}%")

    # 转换为DataFrame
    if all_results:
        df = pd.DataFrame(all_results)
        return df
    else:
        return None


def generate_comparison_report(df, output_file=None):
    """生成对比报告

    Args:
        df: 结果DataFrame
        output_file: 输出文件路径（可选）
    """
    if df is None or len(df) == 0:
        print("没有数据可供分析")
        return

    report = []
    report.append("="*90)
    report.append("VIPER模型性能对比报告")
    report.append("="*90)
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"总模型数: {len(df)}")
    report.append("")

    # 按深度和叶子节点分组统计
    if 'depth' in df.columns and 'leaves' in df.columns:
        report.append("配置分布:")
        report.append(f"  深度范围: {df['depth'].min()} - {df['depth'].max()}")
        report.append(f"  叶子节点范围: {df['leaves'].min()} - {df['leaves'].max()}")
        report.append("")

    # vs Random 最佳模型
    if 'random_win_rate' in df.columns:
        report.append("-" * 90)
        report.append("对战 RANDOM 最佳模型 (按胜率)")
        report.append("-" * 90)

        best_random = df.loc[df['random_win_rate'].idxmax()]
        report.append(f"模型: {best_random.get('basename', 'N/A')}")
        report.append(f"配置: depth={best_random.get('depth', 'N/A')}, "
                     f"leaves={best_random.get('leaves', 'N/A')}")
        report.append(f"胜率: {best_random['random_win_rate']*100:.2f}%")
        report.append(f"平局率: {best_random['random_draw_rate']*100:.2f}%")
        report.append(f"负率: {best_random['random_loss_rate']*100:.2f}%")
        report.append(f"平均奖励: {best_random['random_mean_reward']:.4f}")
        report.append("")

    # vs MinMax 最佳模型
    if 'minmax_draw_rate' in df.columns:
        report.append("-" * 90)
        report.append("对战 MINMAX 最佳模型 (按平局率)")
        report.append("-" * 90)

        best_minmax = df.loc[df['minmax_draw_rate'].idxmax()]
        report.append(f"模型: {best_minmax.get('basename', 'N/A')}")
        report.append(f"配置: depth={best_minmax.get('depth', 'N/A')}, "
                     f"leaves={best_minmax.get('leaves', 'N/A')}")
        report.append(f"胜率: {best_minmax['minmax_win_rate']*100:.2f}%")
        report.append(f"平局率: {best_minmax['minmax_draw_rate']*100:.2f}%")
        report.append(f"负率: {best_minmax['minmax_loss_rate']*100:.2f}%")
        report.append(f"平均奖励: {best_minmax['minmax_mean_reward']:.4f}")
        report.append("")

    # 性能表格
    report.append("-" * 90)
    report.append("所有模型性能汇总")
    report.append("-" * 90)

    # 创建汇总表格
    table_data = []
    for _, row in df.iterrows():
        table_row = {
            'depth': row.get('depth', '?'),
            'leaves': row.get('leaves', '?'),
        }

        if 'random_win_rate' in df.columns:
            table_row['random_win'] = f"{row['random_win_rate']*100:.1f}%"
            table_row['random_draw'] = f"{row['random_draw_rate']*100:.1f}%"

        if 'minmax_win_rate' in df.columns:
            table_row['minmax_win'] = f"{row['minmax_win_rate']*100:.1f}%"
            table_row['minmax_draw'] = f"{row['minmax_draw_rate']*100:.1f}%"

        table_data.append(table_row)

    # 打印表头
    header = "Depth | Leaves | Random (W/D) | MinMax (W/D)"
    report.append(header)
    report.append("-" * len(header))

    # 打印数据行
    for row in table_data:
        line = f"{row['depth']:>5} | {row['leaves']:>6} | "
        if 'random_win' in row:
            line += f"{row['random_win']:>5}/{row['random_draw']:>5} | "
        if 'minmax_win' in row:
            line += f"{row['minmax_win']:>5}/{row['minmax_draw']:>5}"
        report.append(line)

    report.append("")

    # 统计分析
    report.append("-" * 90)
    report.append("统计分析")
    report.append("-" * 90)

    if 'random_win_rate' in df.columns:
        report.append(f"vs Random 平均胜率: {df['random_win_rate'].mean()*100:.2f}% "
                     f"(std: {df['random_win_rate'].std()*100:.2f}%)")

    if 'minmax_draw_rate' in df.columns:
        report.append(f"vs MinMax 平均平局率: {df['minmax_draw_rate'].mean()*100:.2f}% "
                     f"(std: {df['minmax_draw_rate'].std()*100:.2f}%)")

    report.append("")
    report.append("="*90)

    # 输出报告
    report_text = "\n".join(report)
    print(report_text)

    # 保存到文件
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\n✓ 报告已保存到: {output_file}")

    return report_text


def save_results_to_csv(df, output_file):
    """保存结果到CSV文件"""
    if df is not None and len(df) > 0:
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"✓ 结果已保存到CSV: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='分析VIPER并行训练结果',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--models-dir', type=str,
                       default='log/viper_mask_ppo_tictactoe',
                       help='模型目录路径')
    parser.add_argument('--pattern', type=str,
                       default='viper_mask_ppo_tree_*.joblib',
                       help='模型文件名模式')
    parser.add_argument('--n-episodes', type=int, default=1000,
                       help='每个对手的评估局数')
    parser.add_argument('--opponents', type=str, nargs='+',
                       default=['random', 'minmax'],
                       help='对手类型列表')
    parser.add_argument('--output-dir', type=str,
                       default='log/viper_parallel_training',
                       help='输出目录')
    parser.add_argument('--no-evaluate', action='store_true',
                       help='跳过评估，仅使用已有结果')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not args.no_evaluate:
        # 收集所有结果
        print("="*90)
        print("开始收集和评估模型")
        print("="*90)
        print(f"模型目录: {args.models_dir}")
        print(f"文件模式: {args.pattern}")
        print(f"对手: {', '.join(args.opponents)}")
        print(f"评估局数: {args.n_episodes}")
        print()

        df = collect_all_results(
            models_dir=args.models_dir,
            opponents=args.opponents,
            n_episodes=args.n_episodes,
            pattern=args.pattern
        )

        if df is None:
            print("未能收集到结果")
            return

        # 保存原始数据
        csv_file = os.path.join(args.output_dir, f"results_{timestamp}.csv")
        save_results_to_csv(df, csv_file)
    else:
        # 加载已有结果
        csv_files = glob.glob(os.path.join(args.output_dir, "results_*.csv"))
        if not csv_files:
            print(f"未找到结果文件: {args.output_dir}/results_*.csv")
            return

        latest_csv = max(csv_files, key=os.path.getmtime)
        print(f"加载结果文件: {latest_csv}")
        df = pd.read_csv(latest_csv)

    # 生成报告
    print("\n")
    report_file = os.path.join(args.output_dir, f"report_{timestamp}.txt")
    generate_comparison_report(df, report_file)


if __name__ == "__main__":
    main()
