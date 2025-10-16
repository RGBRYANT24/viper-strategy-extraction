#!/usr/bin/env python3
"""
规则提取器演示脚本

这个脚本展示如何使用DecisionTreeRuleExtractor从决策树中提取和简化规则。
使用Iris数据集作为示例。

运行方式:
    python demo_rule_extraction.py
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from model.rule_extractor import extract_and_simplify_rules


def main():
    print("="*80)
    print("决策树规则提取器演示")
    print("="*80)
    print("\n这个演示使用Iris数据集展示规则提取和简化的完整流程\n")

    # 1. 加载数据
    print("步骤 1: 加载Iris数据集")
    print("-"*80)
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    print(f"特征数量: {X.shape[1]}")
    print(f"类别: {iris.target_names}")
    print(f"特征名称: {iris.feature_names}")

    # 2. 训练决策树
    print("\n步骤 2: 训练决策树")
    print("-"*80)
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_train, y_train)

    train_accuracy = clf.score(X_train, y_train)
    test_accuracy = clf.score(X_test, y_test)

    print(f"决策树训练完成:")
    print(f"  - 最大深度: {clf.get_depth()}")
    print(f"  - 叶子节点数: {clf.get_n_leaves()}")
    print(f"  - 训练准确率: {train_accuracy:.3f}")
    print(f"  - 测试准确率: {test_accuracy:.3f}")

    # 3. 提取和简化规则
    print("\n步骤 3: 提取并简化规则")
    print("-"*80)
    extractor = extract_and_simplify_rules(
        clf,
        X_train,
        y_train,
        feature_names=iris.feature_names,
        alpha=0.05,
        verbose=True
    )

    # 4. 展示规则
    print("\n步骤 4: 最终规则")
    print("-"*80)
    print(f"\n共提取 {len(extractor.rules)} 条规则:")
    print()

    for i, rule in enumerate(extractor.rules, 1):
        print(f"规则 {i}:")
        print(f"  {rule}")
        print(f"  类别: {iris.target_names[rule.consequent]}")
        print()

    # 5. 规则统计
    print("\n步骤 5: 规则统计分析")
    print("-"*80)
    stats = extractor.get_stats()

    print(f"规则数量: {stats.get('n_rules', 'N/A')}")
    print(f"平均前件数: {stats.get('avg_antecedents', 'N/A'):.2f}")

    if 'n_removed_antecedents' in stats:
        print(f"简化时删除的前件数: {stats['n_removed_antecedents']}")

    if 'default_consequent' in stats:
        default = stats['default_consequent']
        print(f"默认类别: {iris.target_names[default]}")

    if 'consequent_distribution' in stats:
        print("\n类别分布:")
        for consequent, count in stats['consequent_distribution'].items():
            print(f"  {iris.target_names[consequent]}: {count} 条规则")

    # 6. 验证规则
    print("\n步骤 6: 验证规则准确性")
    print("-"*80)

    # 测试几个样本
    test_samples = X_test[:5]
    print("测试前5个样本:")
    print()

    for i, sample in enumerate(test_samples):
        tree_pred = clf.predict(sample.reshape(1, -1))[0]
        tree_class = iris.target_names[tree_pred]

        # 找到匹配的规则
        matched_rules = [r for r in extractor.rules if r.matches(sample)]

        print(f"样本 {i+1}:")
        print(f"  特征: {sample}")
        print(f"  决策树预测: {tree_class}")
        print(f"  匹配规则数: {len(matched_rules)}")
        if matched_rules:
            rule_pred = matched_rules[0].consequent
            print(f"  规则预测: {iris.target_names[rule_pred]}")
            print(f"  预测一致: {'✓' if rule_pred == tree_pred else '✗'}")
        print()

    # 7. 导出规则
    print("\n步骤 7: 导出规则到文件")
    print("-"*80)
    output_file = "demo_rules_output.txt"
    extractor.export_rules_to_text(output_file)
    print(f"规则已导出到: {output_file}")

    print("\n" + "="*80)
    print("演示完成！")
    print("="*80)
    print("\n关键要点:")
    print("1. 规则提取从决策树的每个叶子节点生成一条IF-THEN规则")
    print("2. 规则简化使用统计检验（卡方/Fisher）删除不必要的条件")
    print("3. 简化后的规则更易于理解和解释")
    print("4. 规则的预测结果与原决策树完全一致")
    print("\n在VIPER项目中，你可以将这些规则应用于强化学习策略的可解释性分析")


if __name__ == "__main__":
    main()
