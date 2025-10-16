from typing import Optional, Tuple, List

import numpy as np
from joblib import load, dump
from sklearn.tree import DecisionTreeClassifier


# Wrapper around our extracted decision tree, mostly so that we can use the sb policy evaluator
class TreeWrapper:
    def __init__(self, tree: DecisionTreeClassifier):
        self.tree = tree
        self._X_train = None
        self._y_train = None
        self._rule_extractor = None

    def predict(
            self,
            observation: np.ndarray,
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        return self.tree.predict(observation), None

    @classmethod
    def load(cls, path: str):
        clf = load(path)
        return TreeWrapper(clf)

    def save(self, path: str):
        print(f"Saving to\t{path}")
        dump(self.tree, path)

    def print_info(self):
        print(f"Max depth:\t{self.tree.get_depth()}")
        print(f"# Leaves:\t{self.tree.get_n_leaves()}")

    def set_training_data(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        设置训练数据，用于规则提取

        Args:
            X_train: 训练数据特征
            y_train: 训练数据标签
        """
        self._X_train = X_train
        self._y_train = y_train
        # 重置规则提取器
        self._rule_extractor = None

    def extract_rules(self, alpha: float = 0.05, verbose: bool = True):
        """
        从决策树中提取并简化规则

        Args:
            alpha: 统计检验显著性水平
            verbose: 是否打印详细信息

        Returns:
            DecisionTreeRuleExtractor实例

        Raises:
            ValueError: 如果未设置训练数据
        """
        if self._X_train is None or self._y_train is None:
            raise ValueError("必须先调用 set_training_data() 设置训练数据才能提取规则")

        from model.rule_extractor import extract_and_simplify_rules

        self._rule_extractor = extract_and_simplify_rules(
            self.tree,
            self._X_train,
            self._y_train,
            alpha=alpha,
            verbose=verbose
        )

        return self._rule_extractor

    def get_rules(self):
        """
        获取已提取的规则

        Returns:
            规则列表，如果还未提取则返回None
        """
        if self._rule_extractor is None:
            return None
        return self._rule_extractor.rules

    def print_rules(self, max_rules: Optional[int] = None):
        """
        打印规则

        Args:
            max_rules: 最多打印多少条规则

        Raises:
            ValueError: 如果还未提取规则
        """
        if self._rule_extractor is None:
            raise ValueError("必须先调用 extract_rules() 提取规则")

        self._rule_extractor.print_rules(max_rules=max_rules)

    def export_rules(self, filepath: str):
        """
        导出规则到文本文件

        Args:
            filepath: 输出文件路径

        Raises:
            ValueError: 如果还未提取规则
        """
        if self._rule_extractor is None:
            raise ValueError("必须先调用 extract_rules() 提取规则")

        self._rule_extractor.export_rules_to_text(filepath)
