"""
决策树规则提取和简化模块

该模块提供从sklearn决策树中提取规则并进行统计学简化的功能。
主要功能：
1. 从决策树中提取IF-THEN规则
2. 使用卡方检验、Yates校正和Fisher精确检验简化规则
3. 消除冗余规则，识别默认规则

作者: VIPER项目组
"""

import numpy as np
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
from typing import List, Tuple, Optional, Dict
import warnings


class Rule:
    """表示一条IF-THEN规则"""

    def __init__(self, antecedents: List[Tuple[int, str, float]],
                 consequent: int,
                 support_count: int = 0):
        """
        初始化规则

        Args:
            antecedents: 前件列表 [(feature_idx, operator, value), ...]
                        operator in ['<=', '>']
            consequent: 后件（预测的类别/动作）
            support_count: 支持该规则的样本数量
        """
        self.antecedents = antecedents
        self.consequent = consequent
        self.support_count = support_count

    def __str__(self) -> str:
        """生成可读的规则字符串"""
        if not self.antecedents:
            return f"DEFAULT: class = {self.consequent}"

        conditions = []
        for feature_idx, operator, value in self.antecedents:
            conditions.append(f"X[{feature_idx}] {operator} {value:.3f}")
        return f"IF {' AND '.join(conditions)} THEN class = {self.consequent} (support={self.support_count})"

    def __repr__(self) -> str:
        return self.__str__()

    def matches(self, X: np.ndarray) -> bool:
        """
        判断样本X是否匹配该规则

        Args:
            X: 样本特征向量

        Returns:
            True if X matches all antecedents, False otherwise
        """
        for feature_idx, operator, value in self.antecedents:
            if operator == '<=':
                if not (X[feature_idx] <= value):
                    return False
            else:  # operator == '>'
                if not (X[feature_idx] > value):
                    return False
        return True

    def to_dict(self) -> Dict:
        """将规则转换为字典格式"""
        return {
            'antecedents': self.antecedents,
            'consequent': self.consequent,
            'support_count': self.support_count,
            'rule_string': str(self)
        }


class DecisionTreeRuleExtractor:
    """决策树规则提取和简化器"""

    def __init__(self, tree_model: DecisionTreeClassifier,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 feature_names: Optional[List[str]] = None,
                 alpha: float = 0.05):
        """
        初始化规则提取器

        Args:
            tree_model: 训练好的sklearn决策树模型
            X_train: 训练数据特征
            y_train: 训练数据标签
            feature_names: 特征名称列表（可选）
            alpha: 统计检验显著性水平（默认0.05）
        """
        self.tree = tree_model
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names
        self.alpha = alpha
        self.rules: List[Rule] = []
        self._extraction_stats = {}

    def extract_rules(self, verbose: bool = False) -> List[Rule]:
        """
        从决策树中提取所有规则

        Args:
            verbose: 是否打印详细信息

        Returns:
            提取的规则列表
        """
        tree_ = self.tree.tree_
        feature = tree_.feature
        threshold = tree_.threshold

        def recurse(node: int, antecedents: List[Tuple[int, str, float]]):
            """递归遍历决策树节点"""
            # 如果是叶节点
            if tree_.feature[node] == -2:
                # 获取该叶节点的类别
                values = tree_.value[node][0]
                consequent = np.argmax(values)
                support_count = int(np.sum(values))

                rule = Rule(antecedents.copy(), consequent, support_count)
                self.rules.append(rule)
            else:
                # 内部节点，继续递归
                feature_idx = feature[node]
                threshold_val = threshold[node]

                # 左子树 (<=)
                left_antecedents = antecedents + [(feature_idx, '<=', threshold_val)]
                recurse(tree_.children_left[node], left_antecedents)

                # 右子树 (>)
                right_antecedents = antecedents + [(feature_idx, '>', threshold_val)]
                recurse(tree_.children_right[node], right_antecedents)

        # 从根节点开始提取
        self.rules = []  # 重置规则列表
        recurse(0, [])

        self._extraction_stats['n_rules'] = len(self.rules)
        self._extraction_stats['avg_antecedents'] = np.mean([len(r.antecedents) for r in self.rules])

        if verbose:
            print(f"提取到 {len(self.rules)} 条规则")
            print(f"平均每条规则有 {self._extraction_stats['avg_antecedents']:.1f} 个前件")

        return self.rules

    def build_contingency_table(self, rule: Rule, antecedent_idx: int) -> np.ndarray:
        """
        为规则构建列联表，用于测试某个前件的独立性

        列联表格式:
                    C1 (符合结论)  C2 (不符合结论)
        R1 (符合前件)    x11            x12
        R2 (不符合前件)   x21            x22

        Args:
            rule: 要测试的规则
            antecedent_idx: 要测试的前件在rule.antecedents中的索引

        Returns:
            2x2列联表 [[x11, x12], [x21, x22]]
        """
        # 创建不包含指定前件的规则
        reduced_antecedents = [ant for i, ant in enumerate(rule.antecedents)
                               if i != antecedent_idx]

        # 初始化列联表计数
        x11, x12, x21, x22 = 0, 0, 0, 0

        test_antecedent = rule.antecedents[antecedent_idx]

        for i in range(len(self.X_train)):
            X = self.X_train[i]
            y = self.y_train[i]

            # 检查是否满足其他前件
            match_others = True
            for feature_idx, operator, value in reduced_antecedents:
                if operator == '<=':
                    if not (X[feature_idx] <= value):
                        match_others = False
                        break
                else:
                    if not (X[feature_idx] > value):
                        match_others = False
                        break

            if not match_others:
                continue

            # 检查是否满足被测试的前件
            feature_idx, operator, value = test_antecedent
            if operator == '<=':
                match_test = X[feature_idx] <= value
            else:
                match_test = X[feature_idx] > value

            # 检查结论是否匹配
            conclusion_match = (y == rule.consequent)

            # 更新列联表
            if match_test and conclusion_match:
                x11 += 1
            elif match_test and not conclusion_match:
                x12 += 1
            elif not match_test and conclusion_match:
                x21 += 1
            else:
                x22 += 1

        return np.array([[x11, x12], [x21, x22]])

    def test_independence(self, contingency_table: np.ndarray) -> Tuple[bool, float, float]:
        """
        对列联表进行独立性检验

        根据期望频数大小自动选择合适的检验方法：
        - 期望频数 >= 10: 使用标准卡方检验
        - 期望频数 >= 5: 使用Yates连续性校正
        - 期望频数 < 5: 使用Fisher精确检验

        Args:
            contingency_table: 2x2列联表

        Returns:
            (is_independent, chi2_stat, p_value)
            - is_independent: 是否独立（True表示可以删除该前件）
            - chi2_stat: 卡方统计量（Fisher检验时为0）
            - p_value: p值
        """
        # 计算边际和
        row_totals = contingency_table.sum(axis=1)
        col_totals = contingency_table.sum(axis=0)
        total = contingency_table.sum()

        if total == 0:
            return True, 0.0, 1.0

        # 计算期望频数
        expected = np.outer(row_totals, col_totals) / total

        # 找出最大期望频数
        max_expected = expected.max()

        # 根据最大期望频数选择检验方法
        if max_expected >= 10:
            # 使用标准卡方检验
            chi2_stat = np.sum((contingency_table - expected) ** 2 / (expected + 1e-10))
        elif max_expected >= 5:
            # 使用Yates连续性校正
            chi2_stat = np.sum(
                (np.abs(contingency_table - expected) - 0.5) ** 2 / (expected + 1e-10)
            )
        else:
            # 使用Fisher精确检验
            try:
                _, p_value = stats.fisher_exact(contingency_table)
                # Fisher检验直接返回p值
                return p_value > self.alpha, 0.0, p_value
            except Exception as e:
                warnings.warn(f"Fisher精确检验失败: {e}，假定不独立")
                return False, 0.0, 0.0

        # 自由度
        df = (contingency_table.shape[0] - 1) * (contingency_table.shape[1] - 1)

        # 临界值
        critical_value = stats.chi2.ppf(1 - self.alpha, df=df)

        # 计算p值
        p_value = 1 - stats.chi2.cdf(chi2_stat, df=df)

        # 如果 chi2 < 临界值，接受原假设（独立）
        is_independent = chi2_stat < critical_value

        # print('is_independent',is_independent, type(is_independent))

        return is_independent, chi2_stat, p_value

    def simplify_rules(self, verbose: bool = False) -> List[Rule]:
        """
        简化所有规则，删除统计学上不必要的前件

        对每条规则的每个前件进行独立性检验，如果某个前件与结论独立
        （即该前件对预测结果没有显著影响），则删除该前件。

        Args:
            verbose: 是否打印详细信息

        Returns:
            简化后的规则列表
        """
        simplified_rules = []
        n_removed_antecedents = 0

        for rule_idx, rule in enumerate(self.rules):
            # 如果规则只有一个前件或没有前件，不能进一步简化
            if len(rule.antecedents) <= 1:
                simplified_rules.append(rule)
                continue

            # 尝试删除每个前件
            antecedents_to_keep = list(range(len(rule.antecedents)))

            for i in range(len(rule.antecedents)):
                # 构建列联表
                contingency_table = self.build_contingency_table(rule, i)

                # 测试独立性
                is_independent, chi2_stat, p_value = self.test_independence(contingency_table)

                # 如果独立，标记为可删除
                if is_independent and i in antecedents_to_keep:
                    antecedents_to_keep.remove(i)
                    n_removed_antecedents += 1
                    if verbose:
                        print(f"  规则 {rule_idx+1}, 删除前件 {i}: {rule.antecedents[i]} "
                              f"(χ²={chi2_stat:.3f}, p={p_value:.3f})")

            # 创建简化后的规则
            if antecedents_to_keep:
                new_antecedents = [rule.antecedents[i] for i in antecedents_to_keep]
                simplified_rule = Rule(new_antecedents, rule.consequent, rule.support_count)
                simplified_rules.append(simplified_rule)
            else:
                # 如果所有前件都被删除，保留原规则（这种情况很少发生）
                if verbose:
                    print(f"  警告: 规则 {rule_idx+1} 的所有前件都被删除，保留原规则")
                simplified_rules.append(rule)

        self.rules = simplified_rules
        self._extraction_stats['n_removed_antecedents'] = n_removed_antecedents

        if verbose:
            print(f"\n简化完成: 删除了 {n_removed_antecedents} 个前件")
            print(f"简化后保留 {len(self.rules)} 条规则")

        return self.rules

    def eliminate_redundant_rules(self, verbose: bool = False) -> Tuple[List[Rule], Optional[int]]:
        """
        消除冗余规则，识别默认规则

        统计每个结论出现的次数，找出最常见的结论作为候选默认规则。

        Args:
            verbose: 是否打印详细信息

        Returns:
            (rules, default_consequent)
            - rules: 保留的规则列表
            - default_consequent: 默认结论（出现最频繁的）
        """
        # 统计每个结论出现的次数
        consequent_counts = Counter([rule.consequent for rule in self.rules])

        default_consequent = None
        if consequent_counts:
            default_consequent = consequent_counts.most_common(1)[0][0]
            default_count = consequent_counts[default_consequent]

            if verbose:
                print(f"\n最常见的结论: {default_consequent} (出现 {default_count} 次，占 {default_count/len(self.rules)*100:.1f}%)")
                print(f"所有结论分布: {dict(consequent_counts)}")

        self._extraction_stats['default_consequent'] = default_consequent
        self._extraction_stats['consequent_distribution'] = dict(consequent_counts)

        return self.rules, default_consequent

    def get_stats(self) -> Dict:
        """获取提取统计信息"""
        return self._extraction_stats.copy()

    def print_rules(self, max_rules: Optional[int] = None):
        """
        打印所有规则

        Args:
            max_rules: 最多打印多少条规则（None表示全部打印）
        """
        n_rules = len(self.rules) if max_rules is None else min(max_rules, len(self.rules))

        print(f"\n{'='*80}")
        print(f"决策树规则 (共 {len(self.rules)} 条，显示前 {n_rules} 条)")
        print(f"{'='*80}")

        for i, rule in enumerate(self.rules[:n_rules], 1):
            print(f"规则 {i:3d}: {rule}")

        if len(self.rules) > n_rules:
            print(f"... (还有 {len(self.rules) - n_rules} 条规则未显示)")

        print(f"{'='*80}\n")

    def export_rules_to_text(self, filepath: str):
        """
        将规则导出到文本文件

        Args:
            filepath: 输出文件路径
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"决策树规则提取结果\n")
            f.write(f"{'='*80}\n\n")

            f.write(f"统计信息:\n")
            for key, value in self._extraction_stats.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\n{'='*80}\n\n")

            f.write(f"规则列表 (共 {len(self.rules)} 条):\n\n")
            for i, rule in enumerate(self.rules, 1):
                f.write(f"规则 {i:3d}: {rule}\n")

        print(f"规则已导出到: {filepath}")


def extract_and_simplify_rules(tree_model: DecisionTreeClassifier,
                                X_train: np.ndarray,
                                y_train: np.ndarray,
                                feature_names: Optional[List[str]] = None,
                                alpha: float = 0.05,
                                verbose: bool = True) -> DecisionTreeRuleExtractor:
    """
    便捷函数：提取并简化决策树规则

    Args:
        tree_model: 训练好的sklearn决策树模型
        X_train: 训练数据特征
        y_train: 训练数据标签
        feature_names: 特征名称列表
        alpha: 显著性水平
        verbose: 是否打印详细信息

    Returns:
        DecisionTreeRuleExtractor实例
    """
    extractor = DecisionTreeRuleExtractor(tree_model, X_train, y_train,
                                         feature_names, alpha)

    if verbose:
        print("\n" + "="*80)
        print("步骤 1: 从决策树提取规则")
        print("="*80)

    extractor.extract_rules(verbose=verbose)

    if verbose:
        print("\n原始规则:")
        extractor.print_rules(max_rules=10)

    if verbose:
        print("\n" + "="*80)
        print("步骤 2: 简化规则（删除不必要的前件）")
        print("="*80)

    extractor.simplify_rules(verbose=verbose)

    if verbose:
        print("\n简化后的规则:")
        extractor.print_rules(max_rules=10)

    if verbose:
        print("\n" + "="*80)
        print("步骤 3: 分析规则分布")
        print("="*80)

    extractor.eliminate_redundant_rules(verbose=verbose)

    return extractor
