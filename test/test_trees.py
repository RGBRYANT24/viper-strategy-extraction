#!/usr/bin/env python
"""
测试VIPER训练的决策树模型对战random、minmax和神经网络对手
"""
import sys
import os
import joblib
import numpy as np
import gymnasium as gym
import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train.viper_mask_ppo import RegressionTreePolicy, evaluate_policy
from sb3_contrib import MaskablePPO


# 决策树模型目录
MODELS_DIR = "log/viper_mask_ppo_tictactoe"

# 所有要测试的决策树模型 (相对于MODELS_DIR)
TREE_MODELS = [
    # "viper_mask_ppo_tree_20251106_225702_iter10_samples100_depth10_leaves50.joblib",
    # "viper_mask_ppo_tree_20251106_225739_iter10_samples4000_depth10_leaves50.joblib",
    "viper_mask_ppo_tree_20251107_001324_iter10_samples50000_depth10_leaves50.joblib",
    "viper_mask_ppo_tree_20251107_003010_iter10_samples50000_depth15_leaves100.joblib",
    "viper_mask_ppo_tree_20251107_003045_iter10_samples50000_depth15_leaves150.joblib",
    "viper_mask_ppo_tree_20251107_003345_iter10_samples50000_depth10_leaves150.joblib",
]


def parse_model_info(filepath):
    """从文件名中解析模型参数信息"""
    basename = os.path.basename(filepath)
    # 示例: viper_mask_ppo_tree_20251106_225702_iter10_samples100_depth10_leaves50.joblib
    parts = basename.replace('.joblib', '').split('_')

    info = {
        'filepath': filepath,
        'basename': basename,
    }

    # 提取参数
    for part in parts:
        if part.startswith('iter'):
            info['iterations'] = part.replace('iter', '')
        elif part.startswith('samples'):
            info['samples'] = part.replace('samples', '')
        elif part.startswith('depth'):
            info['depth'] = part.replace('depth', '')
        elif part.startswith('leaves'):
            info['leaves'] = part.replace('leaves', '')
        elif len(part) == 15 and part[0:8].isdigit() and part[9:15].isdigit():  # timestamp
            info['timestamp'] = part

    return info


def test_single_model(model_path, test_random=True, test_minmax=True, n_episodes=1000):
    """测试单个模型对战不同对手"""
    print(f"\n{'='*90}")
    print(f"测试模型: {os.path.basename(model_path)}")
    print(f"{'='*90}")

    # 加载模型
    try:
        tree = joblib.load(model_path)
        policy = RegressionTreePolicy(tree)
        print(f"✓ 模型加载成功")
        print(f"  树深度: {tree.tree_.max_depth}")
        print(f"  叶子节点数: {tree.tree_.n_leaves}")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return None

    # 测试对战不同对手
    results = {}

    # 1. 对战 Random
    if test_random:
        print(f"\n--- 对战 RANDOM ({n_episodes} 局) ---")
        try:
            result = evaluate_policy(
                policy,
                env_name='TicTacToe-v0',
                opponent_type='random',
                n_episodes=n_episodes
            )
            results['random'] = result

            print(f"  平均奖励: {result['mean_reward']:.4f} ± {result['std_reward']:.4f}")
            print(f"  胜率: {result['win_rate']*100:.2f}% ({result['wins']}/{n_episodes})")
            print(f"  平局率: {result['draw_rate']*100:.2f}% ({result['draws']}/{n_episodes})")
            print(f"  负率: {result['loss_rate']*100:.2f}% ({result['losses']}/{n_episodes})")

        except Exception as e:
            print(f"  ✗ 评估失败: {e}")
            results['random'] = None

    # 2. 对战 MinMax
    if test_minmax:
        print(f"\n--- 对战 MINMAX ({n_episodes} 局) ---")
        try:
            result = evaluate_policy(
                policy,
                env_name='TicTacToe-v0',
                opponent_type='minmax',
                n_episodes=n_episodes
            )
            results['minmax'] = result

            print(f"  平均奖励: {result['mean_reward']:.4f} ± {result['std_reward']:.4f}")
            print(f"  胜率: {result['win_rate']*100:.2f}% ({result['wins']}/{n_episodes})")
            print(f"  平局率: {result['draw_rate']*100:.2f}% ({result['draws']}/{n_episodes})")
            print(f"  负率: {result['loss_rate']*100:.2f}% ({result['losses']}/{n_episodes})")

        except Exception as e:
            print(f"  ✗ 评估失败: {e}")
            results['minmax'] = None

    return results


def test_all_models(test_random=True, test_minmax=True, n_episodes=1000):
    """测试所有模型并生成对比报告"""
    print(f"{'='*90}")
    print("VIPER决策树模型评估工具")
    print(f"{'='*90}")
    print(f"测试配置:")
    print(f"  - 模型目录: {MODELS_DIR}")
    print(f"  - 测试局数: {n_episodes}")
    print(f"  - 对战Random: {'是' if test_random else '否'}")
    print(f"  - 对战MinMax: {'是' if test_minmax else '否'}")
    print()

    # 测试所有模型
    all_results = []
    for i, model_name in enumerate(TREE_MODELS, 1):
        model_path = os.path.join(MODELS_DIR, model_name)

        if not os.path.exists(model_path):
            print(f"\n[{i}/{len(TREE_MODELS)}] ✗ 模型文件不存在: {model_name}")
            print(f"   期望路径: {model_path}")
            continue

        print(f"\n[{i}/{len(TREE_MODELS)}] 测试模型: {model_name}")

        model_info = parse_model_info(model_path)
        results = test_single_model(model_path, test_random, test_minmax, n_episodes)

        if results:
            all_results.append({
                'info': model_info,
                'results': results
            })

    # 生成汇总报告
    print(f"\n\n{'='*90}")
    print("汇总报告")
    print(f"{'='*90}\n")

    # 创建表格
    if test_random and test_minmax:
        print(f"{'模型参数':<45} | {'vs Random':<20} | {'vs MinMax':<20}")
        print(f"{'-'*45}-+-{'-'*20}-+-{'-'*20}")
    elif test_random:
        print(f"{'模型参数':<45} | {'vs Random':<20}")
        print(f"{'-'*45}-+-{'-'*20}")
    elif test_minmax:
        print(f"{'模型参数':<45} | {'vs MinMax':<20}")
        print(f"{'-'*45}-+-{'-'*20}")

    for entry in all_results:
        info = entry['info']
        results = entry['results']

        # 构建参数描述
        params = f"s{info.get('samples', '?'):>5}_d{info.get('depth', '?'):>2}_l{info.get('leaves', '?'):>3}"

        # 获取结果
        parts = [params]

        if test_random:
            if 'random' in results and results['random']:
                r = results['random']
                random_str = f"W:{r['win_rate']*100:4.1f}% D:{r['draw_rate']*100:4.1f}%"
            else:
                random_str = "N/A"
            parts.append(random_str)

        if test_minmax:
            if 'minmax' in results and results['minmax']:
                r = results['minmax']
                minmax_str = f"W:{r['win_rate']*100:4.1f}% D:{r['draw_rate']*100:4.1f}%"
            else:
                minmax_str = "N/A"
            parts.append(minmax_str)

        print(f"{parts[0]:<45} | {' | '.join(parts[1:])}")

    # 找出最佳模型
    print(f"\n{'='*90}")
    print("最佳模型")
    print(f"{'='*90}\n")

    if test_random:
        # 对战random最佳 (胜率最高)
        best_random = max(
            all_results,
            key=lambda x: x['results'].get('random', {}).get('win_rate', 0) if x['results'].get('random') else 0
        )
        if best_random['results'].get('random'):
            r = best_random['results']['random']
            print(f"✓ 对战Random最佳: {best_random['info']['basename']}")
            print(f"  胜率: {r['win_rate']*100:.2f}%, 平均奖励: {r['mean_reward']:.4f}")
            print(f"  参数: samples={best_random['info'].get('samples')}, "
                  f"depth={best_random['info'].get('depth')}, "
                  f"leaves={best_random['info'].get('leaves')}")

    if test_minmax:
        # 对战minmax最佳 (平局率最高)
        best_minmax = max(
            all_results,
            key=lambda x: x['results'].get('minmax', {}).get('draw_rate', 0) if x['results'].get('minmax') else 0
        )
        if best_minmax['results'].get('minmax'):
            r = best_minmax['results']['minmax']
            print(f"\n✓ 对战MinMax最佳: {best_minmax['info']['basename']}")
            print(f"  平局率: {r['draw_rate']*100:.2f}%, 平均奖励: {r['mean_reward']:.4f}")
            print(f"  参数: samples={best_minmax['info'].get('samples')}, "
                  f"depth={best_minmax['info'].get('depth')}, "
                  f"leaves={best_minmax['info'].get('leaves')}")

    # 保存结果到文件
    output_file = os.path.join(MODELS_DIR, "tree_evaluation_results.txt")
    with open(output_file, 'w') as f:
        f.write("VIPER决策树模型评估报告\n")
        f.write("="*90 + "\n\n")
        f.write(f"测试局数: {n_episodes}\n")
        f.write(f"对战对手: ")
        opponents_list = []
        if test_random:
            opponents_list.append("Random")
        if test_minmax:
            opponents_list.append("MinMax")
        f.write(", ".join(opponents_list) + "\n\n")

        for entry in all_results:
            info = entry['info']
            results = entry['results']

            f.write(f"\n模型: {info['basename']}\n")
            f.write(f"{'-'*90}\n")
            f.write(f"参数: iter={info.get('iterations', '?')}, "
                   f"samples={info.get('samples', '?')}, "
                   f"depth={info.get('depth', '?')}, "
                   f"leaves={info.get('leaves', '?')}\n\n")

            for opponent_name, opponent_key in [('Random', 'random'), ('MinMax', 'minmax')]:
                if opponent_key in results and results[opponent_key]:
                    r = results[opponent_key]
                    f.write(f"{opponent_name}:\n")
                    f.write(f"  平均奖励: {r['mean_reward']:.4f} ± {r['std_reward']:.4f}\n")
                    f.write(f"  胜/平/负: {r['wins']}/{r['draws']}/{r['losses']}\n")
                    f.write(f"  胜率: {r['win_rate']*100:.2f}%\n")
                    f.write(f"  平局率: {r['draw_rate']*100:.2f}%\n")
                    f.write(f"  负率: {r['loss_rate']*100:.2f}%\n\n")

    print(f"\n✓ 详细结果已保存到: {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='测试VIPER训练的决策树模型对战random和minmax对手'
    )
    parser.add_argument('--n-episodes', type=int, default=1000,
                       help='每个对手的测试局数 (默认: 1000)')
    parser.add_argument('--no-random', action='store_true',
                       help='不测试对战Random')
    parser.add_argument('--no-minmax', action='store_true',
                       help='不测试对战MinMax')

    args = parser.parse_args()

    test_all_models(
        test_random=not args.no_random,
        test_minmax=not args.no_minmax,
        n_episodes=args.n_episodes
    )


if __name__ == "__main__":
    main()
