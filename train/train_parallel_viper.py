#!/usr/bin/env python
"""
并行训练多个VIPER决策树配置
支持在服务器上同时训练不同的max_depth和max_leaves组合
支持多GPU自动分配
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

# 注意:不在顶层导入train_viper,避免主进程CUDA初始化
# from train.viper_mask_ppo import train_viper


def train_single_config(config):
    """训练单个配置(用于多进程调用)

    Args:
        config: dict包含训练参数
            {
                'oracle_path': str,
                'output_path': str,
                'n_iterations': int,
                'samples_per_iter': int,
                'max_depth': int,
                'max_leaves': int,
                'config_id': int,
                'gpu_id': int or None  # 新增:分配的GPU ID
            }

    Returns:
        dict: 训练结果信息
    """
    # ========== 关键:必须在导入任何CUDA相关库之前设置 ==========
    import os
    import sys
    
    gpu_id = config.get('gpu_id')
    if gpu_id is not None:
        # 为当前进程指定GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    else:
        # 强制使用CPU
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # 清除已导入的torch相关模块,强制重新导入
    modules_to_remove = [key for key in sys.modules.keys() 
                         if 'torch' in key or 'cuda' in key]
    for module in modules_to_remove:
        del sys.modules[module]
    # ==========================================================
    
    # 现在才导入项目依赖
    from train.viper_mask_ppo import train_viper
    
    config_id = config.get('config_id', 'unknown')
    max_depth = config['max_depth']
    max_leaves = config['max_leaves']

    gpu_info = f"GPU {gpu_id}" if gpu_id is not None else "CPU"
    print(f"\n{'='*80}")
    print(f"[配置 {config_id}] 开始训练: depth={max_depth}, leaves={max_leaves} ({gpu_info})")
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
            'error': None,
            'gpu_id': gpu_id  # 记录使用的GPU
        }

        print(f"\n[配置 {config_id}] ✓ 训练成功 ({gpu_info})")

    except Exception as e:
        result = {
            'config_id': config_id,
            'max_depth': max_depth,
            'max_leaves': max_leaves,
            'status': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'gpu_id': gpu_id
        }

        print(f"\n[配置 {config_id}] ✗ 训练失败 ({gpu_info}): {e}")
        print(traceback.format_exc())

    return result


def generate_configs(oracle_path, output_base, n_iterations, samples_per_iter,
                     depth_range, leaves_range, output_dir=None, depth_step=5, leaves_step=25):
    """生成所有配置组合

    Args:
        oracle_path: Oracle模型路径
        output_base: 输出基础路径
        n_iterations: 训练迭代次数
        samples_per_iter: 每轮采样数量
        depth_range: (min_depth, max_depth) 深度范围
        leaves_range: (min_leaves, max_leaves) 叶子节点范围
        output_dir: 输出目录(可选,如果为None则使用train_viper的默认行为)
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
        # 如果指定了output_dir,构建完整的输出路径
        if output_dir:
            output_path = os.path.join(output_dir, output_base)
        else:
            output_path = output_base

        config = {
            'config_id': config_id,
            'oracle_path': oracle_path,
            'output_path': output_path,
            'n_iterations': n_iterations,
            'samples_per_iter': samples_per_iter,
            'max_depth': depth,
            'max_leaves': leaf,
            'gpu_id': None  # 稍后分配
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
                    gpu_info = f"GPU {r.get('gpu_id')}" if r.get('gpu_id') is not None else "CPU"
                    print(f"  - 配置 {r['config_id']}: depth={r['max_depth']}, "
                          f"leaves={r['max_leaves']} ({gpu_info})")
                    print(f"    错误: {r['error']}")


def worker_wrapper(config, results_dict):
    """包装函数,存储结果到共享字典
    
    必须在全局作用域定义以支持pickle序列化
    """
    result = train_single_config(config)
    results_dict[config['config_id']] = result


def main():
    # 设置multiprocessing启动方式为spawn(CUDA要求)
    # 必须在创建任何子进程之前设置
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # 如果已经设置过,忽略错误
    
    parser = argparse.ArgumentParser(
        description='并行训练多个VIPER决策树配置(支持多GPU)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 自动检测GPU并并行训练
  python train_parallel_viper.py --depth-range 5 25 --leaves-range 25 150 --yes

  # 强制使用CPU
  CUDA_VISIBLE_DEVICES='' python train_parallel_viper.py --depth-range 5 15 --leaves-range 25 100 --yes

  # 指定步长和其他参数
  python train_parallel_viper.py --depth-range 5 25 --depth-step 5 \\
                                  --leaves-range 25 150 --leaves-step 25 \\
                                  --n-iterations 10 --samples-per-iter 50000 --yes
        """
    )

    # 基本参数
    parser.add_argument('--oracle-path', type=str,
                       default='log/oracle_TicTacToe_ppo_aggressive.zip',
                       help='Oracle模型路径')
    parser.add_argument('--output-base', type=str,
                       default='viper_mask_ppo_tree',
                       help='输出模型基础名称')
    parser.add_argument('--output-dir', type=str,
                       default=None,
                       help='输出目录(默认: log/viper_parallel_training_TIMESTAMP)')

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
                       help='并行工作进程数 (默认: 自动根据GPU/CPU数量决定)')
    parser.add_argument('--force-cpu', action='store_true',
                       help='强制使用CPU (即使有GPU)')
    parser.add_argument('--dry-run', action='store_true',
                       help='只生成配置不实际训练')
    parser.add_argument('--yes', '-y', action='store_true',
                       help='自动确认,跳过交互式提示(用于nohup后台运行)')

    args = parser.parse_args()

    # 确定输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"log/viper_parallel_training_{timestamp}"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ 输出目录: {output_dir}")

    # 生成配置
    configs = generate_configs(
        oracle_path=args.oracle_path,
        output_base=args.output_base,
        n_iterations=args.n_iterations,
        samples_per_iter=args.samples_per_iter,
        depth_range=tuple(args.depth_range),
        leaves_range=tuple(args.leaves_range),
        output_dir=output_dir,
        depth_step=args.depth_step,
        leaves_step=args.leaves_step
    )

    # 保存配置摘要(在输出目录下)
    log_dir = output_dir
    config_file = os.path.join(log_dir, f"configs_{timestamp}.json")
    save_config_summary(configs, config_file)

    # 打印摘要
    print_summary(configs)

    if args.dry_run:
        print("\n[DRY RUN] 配置已生成,不执行训练")
        return

    # ========== GPU检测和分配 ==========
    # 延迟导入torch,避免主进程CUDA初始化影响子进程
    if args.force_cpu:
        print("\n⚠ 强制使用CPU模式")
        n_gpus = 0
    else:
        try:
            import torch
            n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception as e:
            print(f"\n⚠ GPU检测失败: {e}")
            n_gpus = 0
    
    if n_gpus > 0:
        print(f"\n✓ 检测到 {n_gpus} 块GPU")
        
        # 为每个配置分配GPU (轮流分配)
        for i, config in enumerate(configs):
            config['gpu_id'] = i % n_gpus
        
        # 并行进程数等于GPU数量(每块GPU运行1个进程)
        n_workers = min(n_gpus, len(configs))
        
        print(f"  将启动 {n_workers} 个并行进程 (每块GPU运行1个)")
        
        # 打印分配方案
        print("\nGPU分配方案:")
        gpu_configs = {i: [] for i in range(n_gpus)}
        for config in configs:
            gpu_configs[config['gpu_id']].append(config['config_id'])
        
        for gpu_id, config_ids in gpu_configs.items():
            print(f"  GPU {gpu_id}: 配置 {config_ids}")
    else:
        print("\n⚠ 未检测到GPU,将使用CPU")
        
        # 所有配置都用CPU
        for config in configs:
            config['gpu_id'] = None
        
        # 确定CPU工作进程数
        n_workers = args.n_workers
        if n_workers is None:
            n_workers = max(1, cpu_count() - 1)
            n_workers = min(n_workers, len(configs))  # 不超过配置数
        
        print(f"  将启动 {n_workers} 个CPU并行进程")
    # ====================================

    print(f"\n预计总训练配置数: {len(configs)}")

    # 确认开始(除非使用了 --yes 参数)
    if not args.yes:
        response = input("\n是否开始训练? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("训练已取消")
            return
    else:
        print("\n自动确认模式:直接开始训练")

    # 并行训练
    print("\n" + "="*80)
    print("开始并行训练")
    print("="*80)

    start_time = datetime.now()

    if n_workers == 1:
        # 单进程训练(用于调试)
        results = [train_single_config(config) for config in configs]
    else:
        # 多进程训练 - 手动管理进程以确保GPU分配正确
        from multiprocessing import Process, Manager
        
        manager = Manager()
        results_dict = manager.dict()  # 共享字典存储结果
        
        active_processes = []
        config_idx = 0
        
        while config_idx < len(configs) or active_processes:
            # 启动新进程(不超过n_workers个并发)
            while config_idx < len(configs) and len(active_processes) < n_workers:
                config = configs[config_idx]
                p = Process(target=worker_wrapper, args=(config, results_dict))
                p.start()
                active_processes.append((p, config['config_id']))
                print(f"启动进程: 配置 {config['config_id']} (GPU {config.get('gpu_id', 'CPU')})")
                config_idx += 1
            
            # 检查并移除已完成的进程
            for p, cid in active_processes[:]:
                if not p.is_alive():
                    p.join()
                    active_processes.remove((p, cid))
                    print(f"进程完成: 配置 {cid}")
            
            # 短暂休眠,避免忙等待
            import time
            time.sleep(1)
        
        # 从共享字典中提取结果,按config_id排序
        results = [results_dict[i+1] for i in range(len(configs))]

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