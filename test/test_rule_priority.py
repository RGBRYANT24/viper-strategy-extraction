#!/usr/bin/env python3
"""
测试规则优先级功能

这个脚本演示如何使用新添加的规则优先级功能
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.tree_wrapper import TreeWrapper
from model.rule_extractor import DecisionTreeRuleExtractor
from sb3_contrib import MaskablePPO
import gymnasium as gym


def collect_samples(oracle, env, n_samples=1000):
    """收集样本数据"""
    print(f"收集 {n_samples} 个样本...")

    X_samples = []
    y_samples = []

    obs, _ = env.reset()
    collected = 0

    while collected < n_samples:
        # 使用Oracle预测动作
        mask = (obs == 0).astype(bool)
        action, _ = oracle.predict(obs, deterministic=True, action_masks=mask)

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
    return X, y


def test_rule_priority():
    """测试规则优先级功能"""

    # 配置
    tree_path = "log/ppo_aggressive.joblib"  # 请根据实际情况修改路径
    oracle_path = "log/oracle_TicTacToe_ppo_aggressive.zip"  # 请根据实际情况修改路径

    print("="*80)
    print("测试规则优先级功能")
    print("="*80)

    # 1. 检查文件是否存在
    if not os.path.exists(tree_path):
        print(f"错误: 树文件不存在: {tree_path}")
        print("请修改 tree_path 变量为实际的决策树路径")
        return

    if not os.path.exists(oracle_path):
        print(f"错误: Oracle文件不存在: {oracle_path}")
        print("请修改 oracle_path 变量为实际的Oracle路径")
        return

    # 2. 加载决策树
    print("\n步骤 1: 加载决策树...")
    tree_wrapper = TreeWrapper.load(tree_path)
    tree_wrapper.print_info()

    # 3. 加载Oracle和环境
    print("\n步骤 2: 加载Oracle和环境...")
    env = gym.make('TicTacToe-v0', opponent_type='random')
    oracle = MaskablePPO.load(oracle_path, env=env)
    print("Oracle加载成功")

    # 4. 收集训练数据
    print("\n步骤 3: 收集训练数据...")
    X_train, y_train = collect_samples(oracle, env, n_samples=1000)

    # 5. 创建规则提取器
    print("\n步骤 4: 创建规则提取器...")
    extractor = DecisionTreeRuleExtractor(
        tree_wrapper.tree,
        X_train,
        y_train,
        oracle_model=oracle,
        env=env
    )
    print("规则提取器创建成功")

    # 6. 提取规则
    print("\n步骤 5: 提取规则...")
    extractor.extract_rules(verbose=True)

    print("\n前5条规则（未计算优先级）:")
    for i, rule in enumerate(extractor.rules[:5], 1):
        print(f"  规则 {i}: {rule}")

    # 7. 计算优先级
    print("\n步骤 6: 计算规则优先级...")
    extractor.compute_rule_priorities(verbose=True)

    print("\n前5条规则（已计算优先级）:")
    for i, rule in enumerate(extractor.rules[:5], 1):
        print(f"  规则 {i}: {rule}")

    # 8. 按优先级排序
    print("\n步骤 7: 按优先级排序...")
    extractor.sort_rules_by_priority(descending=True)

    print("\n优先级最高的5条规则:")
    for i, rule in enumerate(extractor.rules[:5], 1):
        print(f"  规则 {i}: {rule}")

    print("\n优先级最低的5条规则:")
    for i, rule in enumerate(extractor.rules[-5:], 1):
        print(f"  规则 {len(extractor.rules)-5+i}: {rule}")

    # 9. 显示统计信息
    print("\n步骤 8: 统计信息")
    print("="*80)
    stats = extractor.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 10. 导出规则
    output_path = "test_rules_with_priority.txt"
    print(f"\n步骤 9: 导出规则到 {output_path}...")
    extractor.export_rules_to_text(output_path)

    print("\n" + "="*80)
    print("测试完成！")
    print("="*80)
    print(f"规则已保存到: {output_path}")
    print(f"总共提取 {len(extractor.rules)} 条规则")
    print(f"优先级范围: [{stats['priority_min']:.4f}, {stats['priority_max']:.4f}]")
    print(f"平均优先级: {stats['priority_mean']:.4f}")

    env.close()


if __name__ == "__main__":
    try:
        test_rule_priority()
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n提示: 请确保:")
        print("  1. 已训练好决策树模型")
        print("  2. 已训练好Oracle模型")
        print("  3. 修改脚本中的 tree_path 和 oracle_path 为实际路径")
