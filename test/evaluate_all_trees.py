#!/usr/bin/env python
"""
批量评估决策树模型
- X-only 模型：只测试先手 (play_as_o_prob=0.0)
- XO-random 模型：测试随机先后手 (play_as_o_prob=0.5)
"""
import sys
import os
import joblib
import numpy as np
import gymnasium as gym
from pathlib import Path
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.viper_mask_ppo import RegressionTreePolicy, evaluate_policy


def parse_model_filename(filename):
    """
    从文件名中解析模型信息
    例如: viper_mask_ppo_tree_20251117_215151_X-only_iter15_samples25000_depth10_leaves150.joblib
    """
    basename = os.path.basename(filename)
    parts = basename.replace('.joblib', '').split('_')

    info = {
        'filename': basename,
        'training_type': None,
        'iterations': None,
        'samples': None,
        'depth': None,
        'leaves': None,
        'timestamp': None
    }

    # 检测训练类型
    if 'X-only' in basename:
        info['training_type'] = 'X-only'
    elif 'XO-random' in basename:
        info['training_type'] = 'XO-random'

    # 提取参数
    for i, part in enumerate(parts):
        if part.startswith('iter'):
            info['iterations'] = int(part.replace('iter', ''))
        elif part.startswith('samples'):
            info['samples'] = int(part.replace('samples', ''))
        elif part.startswith('depth'):
            info['depth'] = int(part.replace('depth', ''))
        elif part.startswith('leaves'):
            info['leaves'] = int(part.replace('leaves', ''))
        elif len(part) == 15 and '_' in basename[basename.index(part)-1:basename.index(part)]:
            # 时间戳格式: YYYYMMDD_HHMMSS
            info['timestamp'] = part

    return info


def get_play_as_o_prob(training_type):
    """根据训练类型返回对应的play_as_o_prob"""
    if training_type == 'X-only':
        return 0.0  # 先手only
    elif training_type == 'XO-random':
        return 0.5  # 随机先后手
    else:
        return 0.5  # 默认随机


def evaluate_single_model(model_path, opponent_type='random', n_episodes=1000):
    """
    评估单个模型

    Args:
        model_path: 模型文件路径
        opponent_type: 'random' 或 'minmax'
        n_episodes: 测试局数

    Returns:
        评估结果字典
    """
    # 解析模型信息
    model_info = parse_model_filename(model_path)

    print(f"\n{'='*80}")
    print(f"评估: {model_info['filename']}")
    print(f"{'='*80}")
    print(f"训练类型: {model_info['training_type']}")
    print(f"参数: iter={model_info['iterations']}, samples={model_info['samples']}, "
          f"depth={model_info['depth']}, leaves={model_info['leaves']}")

    # 加载模型
    try:
        tree = joblib.load(model_path)
        policy = RegressionTreePolicy(tree)
        print(f"✓ 模型加载成功 (实际树深度: {tree.tree_.max_depth}, 叶子数: {tree.tree_.n_leaves})")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return None

    # 确定play_as_o_prob
    play_as_o_prob = get_play_as_o_prob(model_info['training_type'])

    print(f"\n对战 {opponent_type.upper()} (play_as_o_prob={play_as_o_prob}, n_episodes={n_episodes})")

    try:
        result = evaluate_policy(
            policy,
            env_name='TicTacToe-v0',
            opponent_type=opponent_type,
            n_episodes=n_episodes,
            play_as_o_prob=play_as_o_prob
        )

        print(f"  胜: {result['wins']:>4} ({result['win_rate']*100:>5.1f}%)")
        print(f"  平: {result['draws']:>4} ({result['draw_rate']*100:>5.1f}%)")
        print(f"  负: {result['losses']:>4} ({result['loss_rate']*100:>5.1f}%)")
        print(f"  平均奖励: {result['mean_reward']:>6.3f} ± {result['std_reward']:.3f}")

        # 添加模型信息到结果
        result['model_info'] = model_info
        result['opponent'] = opponent_type
        result['play_as_o_prob'] = play_as_o_prob

        return result

    except Exception as e:
        print(f"✗ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def find_all_tree_models(base_dir):
    """
    查找目录中的所有决策树模型文件

    Returns:
        dict: {'X-only': [...], 'XO-random': [...]}
    """
    models = {
        'X-only': [],
        'XO-random': []
    }

    base_path = Path(base_dir)

    for joblib_file in base_path.glob('*.joblib'):
        filename = joblib_file.name

        # 跳过训练日志等非模型文件
        if 'training_log' in filename:
            continue

        model_info = parse_model_filename(filename)

        if model_info['training_type'] == 'X-only':
            models['X-only'].append(str(joblib_file))
        elif model_info['training_type'] == 'XO-random':
            models['XO-random'].append(str(joblib_file))

    # 按文件名排序
    models['X-only'].sort()
    models['XO-random'].sort()

    return models


def evaluate_all_models(base_dir, opponents=['random', 'minmax'], n_episodes=None,
                       n_episodes_random=2000, n_episodes_minmax=500):
    """
    评估所有模型

    Args:
        base_dir: 模型目录
        opponents: 对手类型列表
        n_episodes: 所有对手的测试局数（如果指定，会覆盖下面两个参数）
        n_episodes_random: vs random 的测试局数（默认 2000）
        n_episodes_minmax: vs minmax 的测试局数（默认 500）
    """
    # 如果指定了 n_episodes，则所有对手使用相同局数
    if n_episodes is not None:
        n_episodes_random = n_episodes
        n_episodes_minmax = n_episodes

    print("="*80)
    print("决策树模型批量评估工具")
    print("="*80)
    print(f"模型目录: {base_dir}")
    print(f"对手类型: {', '.join(opponents)}")
    if 'random' in opponents and 'minmax' in opponents:
        print(f"测试局数: random={n_episodes_random}, minmax={n_episodes_minmax}")
    elif 'random' in opponents:
        print(f"测试局数: random={n_episodes_random}")
    elif 'minmax' in opponents:
        print(f"测试局数: minmax={n_episodes_minmax}")
    print()

    # 查找所有模型
    models = find_all_tree_models(base_dir)

    print(f"找到模型:")
    print(f"  X-only: {len(models['X-only'])} 个")
    print(f"  XO-random: {len(models['XO-random'])} 个")
    print()

    # 评估所有模型
    all_results = []

    for training_type in ['X-only', 'XO-random']:
        model_list = models[training_type]

        if not model_list:
            continue

        print(f"\n{'#'*80}")
        print(f"# 评估 {training_type} 模型 ({len(model_list)} 个)")
        print(f"{'#'*80}")

        for i, model_path in enumerate(model_list, 1):
            print(f"\n[{i}/{len(model_list)}]")

            model_results = {}

            for opponent in opponents:
                # 根据对手类型选择测试局数
                if opponent == 'random':
                    episodes = n_episodes_random
                elif opponent == 'minmax':
                    episodes = n_episodes_minmax
                else:
                    episodes = 1000  # 默认值

                result = evaluate_single_model(model_path, opponent, episodes)
                if result:
                    model_results[opponent] = result

            if model_results:
                all_results.append(model_results)

    # 生成汇总报告
    print("\n\n" + "="*80)
    print("汇总报告")
    print("="*80)

    # 按训练类型分组
    results_by_type = {
        'X-only': [],
        'XO-random': []
    }

    for model_results in all_results:
        # 获取第一个对手的结果来确定训练类型
        first_result = next(iter(model_results.values()))
        training_type = first_result['model_info']['training_type']
        results_by_type[training_type].append(model_results)

    # 打印汇总表格
    for training_type in ['X-only', 'XO-random']:
        if not results_by_type[training_type]:
            continue

        print(f"\n{training_type} 模型:")
        print("-"*80)

        # 表头
        header = f"{'深度':>4} {'叶子':>4}"
        for opponent in opponents:
            header += f" | vs {opponent.capitalize():>6} (胜%/平%/负%)"
        print(header)
        print("-"*80)

        # 数据行
        for model_results in results_by_type[training_type]:
            first_result = next(iter(model_results.values()))
            info = first_result['model_info']

            row = f"{info['depth']:>4} {info['leaves']:>4}"

            for opponent in opponents:
                if opponent in model_results:
                    r = model_results[opponent]
                    row += f" | {r['win_rate']*100:>5.1f}/{r['draw_rate']*100:>5.1f}/{r['loss_rate']*100:>5.1f}"
                else:
                    row += f" | {'N/A':>17}"

            print(row)

    # 找出最佳模型
    print("\n" + "="*80)
    print("最佳模型")
    print("="*80)

    for training_type in ['X-only', 'XO-random']:
        if not results_by_type[training_type]:
            continue

        print(f"\n{training_type}:")

        for opponent in opponents:
            # 提取该对手的所有结果
            opponent_results = []
            for model_results in results_by_type[training_type]:
                if opponent in model_results:
                    opponent_results.append(model_results[opponent])

            if not opponent_results:
                continue

            if opponent == 'random':
                # Random: 胜率最高
                best = max(opponent_results, key=lambda x: x['win_rate'])
                print(f"  vs {opponent.capitalize()} 最佳 (胜率最高):")
                print(f"    胜率: {best['win_rate']*100:.1f}% "
                      f"(depth={best['model_info']['depth']}, leaves={best['model_info']['leaves']})")
            elif opponent == 'minmax':
                # MinMax: 平局率最高（理想情况）
                best = max(opponent_results, key=lambda x: x['draw_rate'])
                print(f"  vs {opponent.capitalize()} 最佳 (平局率最高):")
                print(f"    平局率: {best['draw_rate']*100:.1f}% "
                      f"(depth={best['model_info']['depth']}, leaves={best['model_info']['leaves']})")

    # 保存结果到JSON文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(base_dir, f'evaluation_results_{timestamp}.json')

    # 准备保存的数据
    save_data = {
        'timestamp': timestamp,
        'config': {
            'base_dir': base_dir,
            'opponents': opponents,
            'n_episodes_random': n_episodes_random if 'random' in opponents else None,
            'n_episodes_minmax': n_episodes_minmax if 'minmax' in opponents else None
        },
        'results': all_results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 详细结果已保存到: {output_file}")

    # 同时保存一份人类可读的文本报告
    txt_output_file = os.path.join(base_dir, f'evaluation_report_{timestamp}.txt')
    with open(txt_output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("决策树模型评估报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"评估时间: {timestamp}\n")
        f.write(f"模型目录: {base_dir}\n")
        if 'random' in opponents and 'minmax' in opponents:
            f.write(f"测试局数: random={n_episodes_random}, minmax={n_episodes_minmax}\n")
        elif 'random' in opponents:
            f.write(f"测试局数: random={n_episodes_random}\n")
        elif 'minmax' in opponents:
            f.write(f"测试局数: minmax={n_episodes_minmax}\n")
        f.write(f"对手类型: {', '.join(opponents)}\n\n")

        for training_type in ['X-only', 'XO-random']:
            if not results_by_type[training_type]:
                continue

            f.write(f"\n{'='*80}\n")
            f.write(f"{training_type} 模型 ({len(results_by_type[training_type])} 个)\n")
            f.write(f"{'='*80}\n\n")

            for model_results in results_by_type[training_type]:
                first_result = next(iter(model_results.values()))
                info = first_result['model_info']

                f.write(f"\n{info['filename']}\n")
                f.write(f"{'-'*80}\n")
                f.write(f"参数: depth={info['depth']}, leaves={info['leaves']}, "
                       f"samples={info['samples']}, iter={info['iterations']}\n")
                f.write(f"先后手设置: play_as_o_prob={first_result['play_as_o_prob']}\n\n")

                for opponent in opponents:
                    if opponent in model_results:
                        r = model_results[opponent]
                        f.write(f"vs {opponent.capitalize()}:\n")
                        f.write(f"  胜: {r['wins']} ({r['win_rate']*100:.2f}%)\n")
                        f.write(f"  平: {r['draws']} ({r['draw_rate']*100:.2f}%)\n")
                        f.write(f"  负: {r['losses']} ({r['loss_rate']*100:.2f}%)\n")
                        f.write(f"  平均奖励: {r['mean_reward']:.4f} ± {r['std_reward']:.4f}\n\n")

    print(f"✓ 文本报告已保存到: {txt_output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='批量评估决策树模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认设置: random=2000局, minmax=500局
  python evaluate_all_trees.py /path/to/models

  # 自定义每个对手的局数
  python evaluate_all_trees.py /path/to/models --n-random 3000 --n-minmax 300

  # 所有对手使用相同局数
  python evaluate_all_trees.py /path/to/models --n-episodes 1000

  # 只对战random
  python evaluate_all_trees.py /path/to/models --opponents random --n-random 5000
        """
    )

    parser.add_argument('model_dir', type=str,
                       help='模型目录路径')
    parser.add_argument('--opponents', nargs='+',
                       choices=['random', 'minmax'],
                       default=['random', 'minmax'],
                       help='对手类型 (默认: random minmax)')
    parser.add_argument('--n-episodes', type=int, default=None,
                       help='所有对手的测试局数 (会覆盖 --n-random 和 --n-minmax)')
    parser.add_argument('--n-random', type=int, default=2000,
                       help='vs random 的测试局数 (默认: 2000)')
    parser.add_argument('--n-minmax', type=int, default=500,
                       help='vs minmax 的测试局数 (默认: 500)')

    args = parser.parse_args()

    # 检查目录是否存在
    if not os.path.isdir(args.model_dir):
        print(f"错误: 目录不存在: {args.model_dir}")
        return 1

    evaluate_all_models(
        base_dir=args.model_dir,
        opponents=args.opponents,
        n_episodes=args.n_episodes,
        n_episodes_random=args.n_random,
        n_episodes_minmax=args.n_minmax
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
