#!/usr/bin/env python
"""
并行训练多个VIPER决策树配置
支持在服务器上同时训练不同的max_depth和max_leaves组合
"""
import os
import sys
import json
import argparse
import itertools
from multiprocessing import Pool, cpu_count
from datetime import datetime
import traceback

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train.viper_mask_ppo import train_viper


def train_single_config(config):
    """训练单个配置（用于多进程调用）

    Args:
        config: dict包含训练参数
            {
                'oracle_path': str,
                'output_path': str,
                'n_iterations': int,
                'samples_per_iter': int,
                'max_depth': int,
                'max_leaves': int,
                'config_id': int
            }

    Returns:
        dict: 训练结果信息
    """
    config_id = config.get('config_id', 'unknown')
    max_depth = config['max_depth']
    max_leaves = config['max_leaves']

    print(f"\n{'='*80}")
    print(f"[配置 {config_id}] 开始训练: depth={max_depth}, leaves={max_leaves}")
    print(f"{'='*80}\n")

    try:
        # 调用训练函数
        train_viper(
            oracle_path=config['oracle_path'],
            output_path=config['output_path'],
            n_iterations=config['n_iterations'],
            samples_per_iter=config['samples_per_iter'],
            max_depth=max_depth,
            max_leaves=max_leaves
        )

        result = {
            'config_id': config_id,
            'max_depth': max_depth,
            'max_leaves': max_leaves,
            'status': 'success',
            'error': None
        }

        print(f"\n[配置 {config_id}] ✓ 训练成功")

    except Exception as e:
        result = {
            'config_id': config_id,
            'max_depth': max_depth,
            'max_leaves': max_leaves,
            'status': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc()
        }

        print(f"\n[配置 {config_id}] ✗ 训练失败: {e}")
        print(traceback.format_exc())

    return result


def generate_configs(oracle_path, output_base, n_iterations, samples_per_iter,
                     depth_range, leaves_range, depth_step=5, leaves_step=25):
    """生成所有配置组合

    Args:
        oracle_path: Oracle模型路径
        output_base: 输出基础路径
        n_iterations: 训练迭代次数
        samples_per_iter: 每轮采样数量
        depth_range: (min_depth, max_depth) 深度范围
        leaves_range: (min_leaves, max_leaves) 叶子节点范围
        depth_step: 深度步长
        leaves_step: 叶子节点步长

    Returns:
        list: 配置字典列表
    """
    min_depth, max_depth = depth_range
    min_leaves, max_leaves = leaves_range

    depths = list(range(min_depth, max_depth + 1, depth_step))
    leaves = list(range(min_leaves, max_leaves + 1, leaves_step))

    configs = []
    config_id = 1

    for depth, leaf in itertools.product(depths, leaves):
        config = {
            'config_id': config_id,
            'oracle_path': oracle_path,
            'output_path': output_base,
            'n_iterations': n_iterations,
            'samples_per_iter': samples_per_iter,
            'max_depth': depth,
            'max_leaves': leaf
        }
        configs.append(config)
        config_id += 1

    return configs


def save_config_summary(configs, output_file):
    """保存配置摘要到JSON文件"""
    summary = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_configs': len(configs),
        'configs': configs
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"✓ 配置摘要已保存: {output_file}")


def save_results_summary(results, output_file):
    """保存训练结果摘要"""
    summary = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_configs': len(results),
        'success_count': sum(1 for r in results if r['status'] == 'success'),
        'failed_count': sum(1 for r in results if r['status'] == 'failed'),
        'results': results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"✓ 结果摘要已保存: {output_file}")


def print_summary(configs, results=None):
    """打印配置和结果摘要"""
    print("\n" + "="*80)
    print("训练配置摘要")
    print("="*80)
    print(f"总配置数: {len(configs)}")

    # 提取深度和叶子节点范围
    depths = sorted(set(c['max_depth'] for c in configs))
    leaves = sorted(set(c['max_leaves'] for c in configs))

    print(f"深度范围: {depths}")
    print(f"叶子节点范围: {leaves}")
    print(f"每个配置的迭代次数: {configs[0]['n_iterations']}")
    print(f"每轮采样数: {configs[0]['samples_per_iter']}")

    if results:
        print("\n" + "="*80)
        print("训练结果摘要")
        print("="*80)
        success = sum(1 for r in results if r['status'] == 'success')
        failed = sum(1 for r in results if r['status'] == 'failed')
        print(f"成功: {success}/{len(results)}")
        print(f"失败: {failed}/{len(results)}")

        if failed > 0:
            print("\n失败的配置:")
            for r in results:
                if r['status'] == 'failed':
                    print(f"  - 配置 {r['config_id']}: depth={r['max_depth']}, "
                          f"leaves={r['max_leaves']}")
                    print(f"    错误: {r['error']}")


def main():
    parser = argparse.ArgumentParser(
        description='并行训练多个VIPER决策树配置',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 训练深度5-25（步长5），叶子节点25-150（步长25）的所有组合
  python train_parallel_viper.py --depth-range 5 25 --leaves-range 25 150

  # 使用4个进程并行训练
  python train_parallel_viper.py --depth-range 5 15 --leaves-range 25 100 --n-workers 4

  # 指定步长和其他参数
  python train_parallel_viper.py --depth-range 5 25 --depth-step 5 \\
                                  --leaves-range 25 150 --leaves-step 25 \\
                                  --n-iterations 10 --samples-per-iter 50000
        """
    )

    # 基本参数
    parser.add_argument('--oracle-path', type=str,
                       default='log/oracle_TicTacToe_ppo_aggressive.zip',
                       help='Oracle模型路径')
    parser.add_argument('--output-base', type=str,
                       default='viper_mask_ppo_tree',
                       help='输出模型基础名称')

    # 配置范围
    parser.add_argument('--depth-range', type=int, nargs=2, required=True,
                       metavar=('MIN', 'MAX'),
                       help='决策树深度范围 (例如: 5 25)')
    parser.add_argument('--depth-step', type=int, default=5,
                       help='深度步长 (默认: 5)')
    parser.add_argument('--leaves-range', type=int, nargs=2, required=True,
                       metavar=('MIN', 'MAX'),
                       help='叶子节点数量范围 (例如: 25 150)')
    parser.add_argument('--leaves-step', type=int, default=25,
                       help='叶子节点步长 (默认: 25)')

    # 训练参数
    parser.add_argument('--n-iterations', type=int, default=10,
                       help='每个配置的VIPER迭代次数 (默认: 10)')
    parser.add_argument('--samples-per-iter', type=int, default=50000,
                       help='每轮采样数量 (默认: 50000)')

    # 并行参数
    parser.add_argument('--n-workers', type=int, default=None,
                       help='并行工作进程数 (默认: CPU核心数-1)')
    parser.add_argument('--dry-run', action='store_true',
                       help='只生成配置不实际训练')
    parser.add_argument('--yes', '-y', action='store_true',
                       help='自动确认，跳过交互式提示（用于nohup后台运行）')

    args = parser.parse_args()

    # 生成配置
    configs = generate_configs(
        oracle_path=args.oracle_path,
        output_base=args.output_base,
        n_iterations=args.n_iterations,
        samples_per_iter=args.samples_per_iter,
        depth_range=tuple(args.depth_range),
        leaves_range=tuple(args.leaves_range),
        depth_step=args.depth_step,
        leaves_step=args.leaves_step
    )

    # 保存配置摘要
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "log/viper_parallel_training"
    os.makedirs(log_dir, exist_ok=True)

    config_file = os.path.join(log_dir, f"configs_{timestamp}.json")
    save_config_summary(configs, config_file)

    # 打印摘要
    print_summary(configs)

    if args.dry_run:
        print("\n[DRY RUN] 配置已生成，不执行训练")
        return

    # 确定工作进程数
    n_workers = args.n_workers
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    print(f"\n将使用 {n_workers} 个并行进程进行训练")
    print(f"预计总训练配置数: {len(configs)}")

    # 确认开始（除非使用了 --yes 参数）
    if not args.yes:
        response = input("\n是否开始训练? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("训练已取消")
            return
    else:
        print("\n自动确认模式：直接开始训练")

    # 并行训练
    print("\n" + "="*80)
    print("开始并行训练")
    print("="*80)

    start_time = datetime.now()

    if n_workers == 1:
        # 单进程训练（用于调试）
        results = [train_single_config(config) for config in configs]
    else:
        # 多进程训练
        with Pool(processes=n_workers) as pool:
            results = pool.map(train_single_config, configs)

    end_time = datetime.now()
    duration = end_time - start_time

    # 保存结果
    result_file = os.path.join(log_dir, f"results_{timestamp}.json")
    save_results_summary(results, result_file)

    # 打印最终摘要
    print("\n" + "="*80)
    print("所有训练完成")
    print("="*80)
    print(f"总耗时: {duration}")
    print_summary(configs, results)

    print(f"\n配置文件: {config_file}")
    print(f"结果文件: {result_file}")


if __name__ == "__main__":
    main()
