"""
分析 VIPER 训练中的状态覆盖率
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gym_env
import gymnasium as gym
import numpy as np
from collections import defaultdict


def state_to_key(obs):
    """将状态转换为唯一的键（用于去重）"""
    # 将 float32 数组转换为 tuple（可哈希）
    return tuple(obs.flatten())


def sample_and_analyze(env, n_samples=10000):
    """采样并分析状态覆盖率"""
    print(f"采样 {n_samples} 个样本，分析状态分布...")

    unique_states = set()
    state_counts = defaultdict(int)

    obs, _ = env.reset()
    sample_count = 0

    while sample_count < n_samples:
        # 随机选择合法动作
        legal_actions = np.where(obs == 0)[0]
        if len(legal_actions) == 0:
            obs, _ = env.reset()
            continue

        action = np.random.choice(legal_actions)

        # 记录状态
        state_key = state_to_key(obs)
        unique_states.add(state_key)
        state_counts[state_key] += 1
        sample_count += 1

        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if done:
            obs, _ = env.reset()

    # 统计分析
    n_unique = len(unique_states)
    counts = list(state_counts.values())
    avg_visits = np.mean(counts)
    max_visits = max(counts)
    min_visits = min(counts)

    print()
    print("=" * 70)
    print("状态覆盖分析")
    print("=" * 70)
    print(f"总采样数: {n_samples:,}")
    print(f"唯一状态数: {n_unique:,}")
    print(f"平均每个状态被访问: {avg_visits:.2f} 次")
    print(f"最多访问次数: {max_visits}")
    print(f"最少访问次数: {min_visits}")
    print(f"覆盖率: {n_unique / 5478 * 100:.1f}% (假设总共 5,478 个有效状态)")

    # 访问次数分布
    visit_distribution = defaultdict(int)
    for count in counts:
        if count == 1:
            visit_distribution['1次'] += 1
        elif count <= 5:
            visit_distribution['2-5次'] += 1
        elif count <= 10:
            visit_distribution['6-10次'] += 1
        elif count <= 20:
            visit_distribution['11-20次'] += 1
        else:
            visit_distribution['20次以上'] += 1

    print()
    print("访问次数分布:")
    for key in ['1次', '2-5次', '6-10次', '11-20次', '20次以上']:
        if key in visit_distribution:
            count = visit_distribution[key]
            pct = count / n_unique * 100
            print(f"  {key:8s}: {count:5d} 个状态 ({pct:5.1f}%)")

    print("=" * 70)

    return {
        'n_samples': n_samples,
        'n_unique': n_unique,
        'avg_visits': avg_visits,
        'coverage_rate': n_unique / 5478
    }


def compare_sample_sizes():
    """比较不同采样量的状态覆盖率"""
    print()
    print("=" * 70)
    print("比较不同采样量的状态覆盖率")
    print("=" * 70)
    print()

    sample_sizes = [1000, 5000, 10000, 20000, 50000, 100000]

    results = []
    for n_samples in sample_sizes:
        env = gym.make('TicTacToe-v0', opponent_type='random', play_as_o_prob=0.5)
        result = sample_and_analyze(env, n_samples)
        results.append(result)
        env.close()
        print()

    # 总结
    print("=" * 70)
    print("总结")
    print("=" * 70)
    print(f"{'采样量':<12} {'唯一状态':<12} {'平均访问':<12} {'覆盖率':<12}")
    print("-" * 70)
    for result in results:
        print(f"{result['n_samples']:<12,} "
              f"{result['n_unique']:<12,} "
              f"{result['avg_visits']:<12.2f} "
              f"{result['coverage_rate']*100:<12.1f}%")
    print("=" * 70)

    # 建议
    print()
    print("💡 建议:")

    # 找到覆盖率达到 90% 的最小采样量
    for result in results:
        if result['coverage_rate'] >= 0.9:
            print(f"  - 采样 {result['n_samples']:,} 个样本即可覆盖 90% 以上的状态")
            print(f"  - 继续增加采样量的边际收益很小")
            break
    else:
        print(f"  - 建议采样至少 100,000 个样本以获得良好覆盖")

    print()


def analyze_first_vs_second_player():
    """分析先手和后手的状态分布差异"""
    print()
    print("=" * 70)
    print("分析先手和后手的状态差异")
    print("=" * 70)
    print()

    n_samples = 20000

    # 先手状态
    env_first = gym.make('TicTacToe-v0', opponent_type='random', play_as_o_prob=0.0)
    print("采样先手状态...")
    states_first = set()
    obs, _ = env_first.reset()
    for _ in range(n_samples):
        legal_actions = np.where(obs == 0)[0]
        if len(legal_actions) == 0:
            obs, _ = env_first.reset()
            continue
        action = np.random.choice(legal_actions)
        states_first.add(state_to_key(obs))
        obs, reward, terminated, truncated, _ = env_first.step(action)
        if terminated or truncated:
            obs, _ = env_first.reset()
    env_first.close()

    # 后手状态
    env_second = gym.make('TicTacToe-v0', opponent_type='random', play_as_o_prob=1.0)
    print("采样后手状态...")
    states_second = set()
    obs, _ = env_second.reset()
    for _ in range(n_samples):
        legal_actions = np.where(obs == 0)[0]
        if len(legal_actions) == 0:
            obs, _ = env_second.reset()
            continue
        action = np.random.choice(legal_actions)
        states_second.add(state_to_key(obs))
        obs, reward, terminated, truncated, _ = env_second.step(action)
        if terminated or truncated:
            obs, _ = env_second.reset()
    env_second.close()

    # 分析
    overlap = states_first & states_second
    only_first = states_first - states_second
    only_second = states_second - states_first

    print()
    print("结果:")
    print(f"  先手唯一状态: {len(states_first):,}")
    print(f"  后手唯一状态: {len(states_second):,}")
    print(f"  共同状态: {len(overlap):,}")
    print(f"  仅先手状态: {len(only_first):,}")
    print(f"  仅后手状态: {len(only_second):,}")

    total_unique = len(states_first | states_second)
    print(f"  总唯一状态（先手+后手）: {total_unique:,}")

    print()
    print("💡 结论:")
    if len(overlap) > len(states_first) * 0.8:
        print("  - 先手和后手的状态空间高度重叠（视角转换有效）")
        print("  - 可以使用较小的采样量")
    else:
        print("  - 先手和后手的状态空间差异较大")
        print("  - 需要更多采样以覆盖两种情况")

    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='分析状态覆盖率')
    parser.add_argument('--mode', type=str, default='compare',
                       choices=['single', 'compare', 'first-vs-second', 'all'],
                       help='运行模式')
    parser.add_argument('--n-samples', type=int, default=10000,
                       help='采样数量（single 模式）')

    args = parser.parse_args()

    if args.mode == 'single':
        env = gym.make('TicTacToe-v0', opponent_type='random', play_as_o_prob=0.5)
        sample_and_analyze(env, args.n_samples)
        env.close()

    elif args.mode == 'compare':
        compare_sample_sizes()

    elif args.mode == 'first-vs-second':
        analyze_first_vs_second_player()

    elif args.mode == 'all':
        compare_sample_sizes()
        analyze_first_vs_second_player()
