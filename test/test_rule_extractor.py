"""
决策树规则提取器的单元测试

测试DecisionTreeRuleExtractor类的各项功能
"""

import unittest
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split

import sys
import os
# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.rule_extractor import Rule, DecisionTreeRuleExtractor, extract_and_simplify_rules


class TestRule(unittest.TestCase):
    """测试Rule类"""

    def test_rule_creation(self):
        """测试规则创建"""
        antecedents = [(0, '<=', 5.5), (1, '>', 3.0)]
        consequent = 1
        support_count = 42

        rule = Rule(antecedents, consequent, support_count)

        self.assertEqual(rule.antecedents, antecedents)
        self.assertEqual(rule.consequent, consequent)
        self.assertEqual(rule.support_count, support_count)

    def test_rule_str(self):
        """测试规则字符串表示"""
        antecedents = [(0, '<=', 5.5), (1, '>', 3.0)]
        rule = Rule(antecedents, 1, 42)

        rule_str = str(rule)
        self.assertIn("IF", rule_str)
        self.assertIn("THEN", rule_str)
        self.assertIn("X[0] <= 5.500", rule_str)
        self.assertIn("X[1] > 3.000", rule_str)
        self.assertIn("class = 1", rule_str)
        self.assertIn("support=42", rule_str)

    def test_rule_matches(self):
        """测试规则匹配功能"""
        antecedents = [(0, '<=', 5.5), (1, '>', 3.0)]
        rule = Rule(antecedents, 1)

        # 应该匹配的样本
        X_match = np.array([5.0, 3.5, 0, 0])
        self.assertTrue(rule.matches(X_match))

        # 不应该匹配的样本（第一个条件不满足）
        X_no_match_1 = np.array([6.0, 3.5, 0, 0])
        self.assertFalse(rule.matches(X_no_match_1))

        # 不应该匹配的样本（第二个条件不满足）
        X_no_match_2 = np.array([5.0, 2.5, 0, 0])
        self.assertFalse(rule.matches(X_no_match_2))

    def test_rule_to_dict(self):
        """测试规则转字典"""
        antecedents = [(0, '<=', 5.5)]
        rule = Rule(antecedents, 1, 10)

        rule_dict = rule.to_dict()
        self.assertIn('antecedents', rule_dict)
        self.assertIn('consequent', rule_dict)
        self.assertIn('support_count', rule_dict)
        self.assertIn('rule_string', rule_dict)
        self.assertEqual(rule_dict['consequent'], 1)
        self.assertEqual(rule_dict['support_count'], 10)

    def test_empty_rule(self):
        """测试空规则（默认规则）"""
        rule = Rule([], 0, 100)
        rule_str = str(rule)
        self.assertIn("DEFAULT", rule_str)


class TestDecisionTreeRuleExtractor(unittest.TestCase):
    """测试DecisionTreeRuleExtractor类"""

    def setUp(self):
        """设置测试数据"""
        # 使用Iris数据集
        iris = load_iris()
        self.X, self.y = iris.data, iris.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42
        )

        # 训练一个简单的决策树
        self.clf = DecisionTreeClassifier(max_depth=3, random_state=42)
        self.clf.fit(self.X_train, self.y_train)

        # 创建规则提取器
        self.extractor = DecisionTreeRuleExtractor(
            self.clf, self.X_train, self.y_train,
            feature_names=iris.feature_names
        )

    def test_extract_rules(self):
        """测试规则提取"""
        rules = self.extractor.extract_rules(verbose=False)

        # 应该提取到规则
        self.assertGreater(len(rules), 0)

        # 每条规则应该有正确的结构
        for rule in rules:
            self.assertIsInstance(rule, Rule)
            self.assertIsInstance(rule.antecedents, list)
            self.assertIsInstance(rule.consequent, (int, np.integer))
            self.assertIsInstance(rule.support_count, (int, np.integer))

        # 规则数量应该等于叶子节点数量
        self.assertEqual(len(rules), self.clf.get_n_leaves())

    def test_extract_rules_statistics(self):
        """测试规则提取统计信息"""
        self.extractor.extract_rules(verbose=False)
        stats = self.extractor.get_stats()

        self.assertIn('n_rules', stats)
        self.assertIn('avg_antecedents', stats)
        self.assertGreater(stats['n_rules'], 0)

    def test_build_contingency_table(self):
        """测试列联表构建"""
        self.extractor.extract_rules(verbose=False)

        # 取第一条有多个前件的规则
        rule = None
        for r in self.extractor.rules:
            if len(r.antecedents) >= 2:
                rule = r
                break

        if rule is not None:
            # 构建列联表
            table = self.extractor.build_contingency_table(rule, 0)

            # 检查列联表形状
            self.assertEqual(table.shape, (2, 2))

            # 列联表应该都是非负整数
            self.assertTrue(np.all(table >= 0))

            # 列联表的总和不应该超过训练集大小
            self.assertLessEqual(table.sum(), len(self.X_train))

    def test_independence_test(self):
        """测试独立性检验"""
        # 创建一个已知的列联表
        table = np.array([[50, 10], [10, 50]])

        is_independent, chi2_stat, p_value = self.extractor.test_independence(table)

        # 检查返回值类型
        self.assertIsInstance(is_independent, np.bool_)
        self.assertIsInstance(chi2_stat, (float, np.floating))
        self.assertIsInstance(p_value, (float, np.floating))

        # p值应该在[0, 1]之间
        self.assertGreaterEqual(p_value, 0)
        self.assertLessEqual(p_value, 1)

    def test_independence_test_edge_cases(self):
        """测试独立性检验的边界情况"""
        # 空表
        table_zero = np.array([[0, 0], [0, 0]])
        is_independent, _, p_value = self.extractor.test_independence(table_zero)
        self.assertTrue(is_independent)
        self.assertEqual(p_value, 1.0)

        # 完全独立的表
        table_independent = np.array([[25, 25], [25, 25]])
        is_independent, _, _ = self.extractor.test_independence(table_independent)
        self.assertTrue(is_independent)

    def test_simplify_rules(self):
        """测试规则简化"""
        self.extractor.extract_rules(verbose=False)
        original_rules = self.extractor.rules.copy()

        simplified_rules = self.extractor.simplify_rules(verbose=False)

        # 应该返回规则列表
        self.assertIsInstance(simplified_rules, list)
        self.assertGreater(len(simplified_rules), 0)

        # 简化后规则数量应该相同（只是前件可能减少）
        self.assertEqual(len(simplified_rules), len(original_rules))

    def test_eliminate_redundant_rules(self):
        """测试冗余规则消除"""
        self.extractor.extract_rules(verbose=False)

        rules, default_consequent = self.extractor.eliminate_redundant_rules(verbose=False)

        # 应该返回规则和默认结论
        self.assertIsInstance(rules, list)
        self.assertIsNotNone(default_consequent)

        # 默认结论应该是一个有效的类别
        self.assertIn(default_consequent, self.y_train)

        # 统计信息应该包含默认结论
        stats = self.extractor.get_stats()
        self.assertIn('default_consequent', stats)
        self.assertIn('consequent_distribution', stats)

    def test_print_rules(self):
        """测试规则打印（不产生错误）"""
        self.extractor.extract_rules(verbose=False)

        # 应该不抛出异常
        try:
            self.extractor.print_rules(max_rules=5)
        except Exception as e:
            self.fail(f"print_rules() raised {type(e).__name__}: {e}")

    def test_export_rules(self):
        """测试规则导出"""
        import tempfile
        import os

        self.extractor.extract_rules(verbose=False)

        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            temp_path = f.name

        try:
            self.extractor.export_rules_to_text(temp_path)

            # 检查文件是否存在
            self.assertTrue(os.path.exists(temp_path))

            # 检查文件是否有内容
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self.assertGreater(len(content), 0)
                self.assertIn('规则', content)

        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestExtractAndSimplifyRules(unittest.TestCase):
    """测试便捷函数"""

    def test_extract_and_simplify_rules(self):
        """测试一站式规则提取和简化"""
        # 创建简单数据集
        X, y = make_classification(n_samples=200, n_features=4, n_informative=3,
                                   n_redundant=1, n_classes=3, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # 训练决策树
        clf = DecisionTreeClassifier(max_depth=4, random_state=42)
        clf.fit(X_train, y_train)

        # 提取并简化规则
        extractor = extract_and_simplify_rules(
            clf, X_train, y_train,
            alpha=0.05,
            verbose=False
        )

        # 检查返回的提取器
        self.assertIsInstance(extractor, DecisionTreeRuleExtractor)
        self.assertGreater(len(extractor.rules), 0)

        # 检查统计信息
        stats = extractor.get_stats()
        self.assertIn('n_rules', stats)
        self.assertIn('default_consequent', stats)


class TestRuleExtractorWithSmallTree(unittest.TestCase):
    """测试小决策树的规则提取"""

    def test_single_node_tree(self):
        """测试单节点树（只有根节点）"""
        # 创建只有一个类别的数据
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 0, 0])

        clf = DecisionTreeClassifier(max_depth=1)
        clf.fit(X, y)

        extractor = DecisionTreeRuleExtractor(clf, X, y)
        rules = extractor.extract_rules(verbose=False)

        # 应该只有一条规则（叶子节点）
        self.assertEqual(len(rules), 1)
        # 这条规则可能有0个或1个前件
        self.assertLessEqual(len(rules[0].antecedents), 1)

    def test_two_class_simple_tree(self):
        """测试简单的二分类树"""
        # 创建可线性分离的数据
        X = np.array([[1], [2], [3], [4]])
        y = np.array([0, 0, 1, 1])

        clf = DecisionTreeClassifier(max_depth=2, random_state=42)
        clf.fit(X, y)

        extractor = DecisionTreeRuleExtractor(clf, X, y)
        rules = extractor.extract_rules(verbose=False)

        # 应该有叶子节点数量的规则
        self.assertEqual(len(rules), clf.get_n_leaves())

        # 每条规则的结论应该是0或1
        for rule in rules:
            self.assertIn(rule.consequent, [0, 1])


class TestRuleMatchingAccuracy(unittest.TestCase):
    """测试规则的预测准确性"""

    def test_rules_match_tree_predictions(self):
        """测试提取的规则是否与原决策树的预测一致"""
        # 创建数据
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # 训练树
        clf = DecisionTreeClassifier(max_depth=3, random_state=42)
        clf.fit(X_train, y_train)

        # 提取规则
        extractor = DecisionTreeRuleExtractor(clf, X_train, y_train)
        extractor.extract_rules(verbose=False)

        # 对测试集中的每个样本，检查规则匹配
        for X_sample in X_test[:10]:  # 测试前10个样本
            # 获取决策树的预测
            tree_pred = clf.predict(X_sample.reshape(1, -1))[0]

            # 找到匹配的规则
            matched_rules = [rule for rule in extractor.rules if rule.matches(X_sample)]

            # 应该至少有一条规则匹配
            self.assertGreater(len(matched_rules), 0,
                             f"No rule matches sample {X_sample}")

            # 匹配的规则的结论应该与树的预测一致
            for rule in matched_rules:
                self.assertEqual(rule.consequent, tree_pred,
                               f"Rule consequent {rule.consequent} != tree prediction {tree_pred}")


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加所有测试
    suite.addTests(loader.loadTestsFromTestCase(TestRule))
    suite.addTests(loader.loadTestsFromTestCase(TestDecisionTreeRuleExtractor))
    suite.addTests(loader.loadTestsFromTestCase(TestExtractAndSimplifyRules))
    suite.addTests(loader.loadTestsFromTestCase(TestRuleExtractorWithSmallTree))
    suite.addTests(loader.loadTestsFromTestCase(TestRuleMatchingAccuracy))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    result = run_tests()
    # 返回退出码
    exit(0 if result.wasSuccessful() else 1)
