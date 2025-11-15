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
                 consequent,
                 support_count: int = 0,
                 priority: float = 0.0,
                 output_vector: Optional[np.ndarray] = None):
        """
        初始化规则

        Args:
            antecedents: 前件列表 [(feature_idx, operator, value), ...]
                        operator in ['<=', '>']
            consequent: 后件（对于分类树是int类别，对于回归树是np.ndarray向量）
            support_count: 支持该规则的样本数量
            priority: 规则的优先级（状态重要性），数值越大越重要
            output_vector: 输出向量（对于回归树，存储完整的预测向量，如9维logits）
        """
        self.antecedents = antecedents
        self.consequent = consequent
        self.support_count = support_count
        self.priority = priority
        self.output_vector = output_vector

        # 如果是回归输出，自动设置output_vector
        if output_vector is None and isinstance(consequent, np.ndarray):
            self.output_vector = consequent.copy()

    def __str__(self) -> str:
        """生成可读的规则字符串"""
        if not self.antecedents:
            if isinstance(self.consequent, np.ndarray):
                return f"DEFAULT: output_vector = {self.consequent}"
            return f"DEFAULT: class = {self.consequent}"

        conditions = []
        for feature_idx, operator, value in self.antecedents:
            conditions.append(f"X[{feature_idx}] {operator} {value:.3f}")

        priority_str = f", priority={self.priority:.4f}" if self.priority > 0 else ""

        # 根据consequent类型决定输出格式
        if isinstance(self.consequent, np.ndarray):
            # 回归树输出：显示向量
            action = np.argmax(self.consequent)
            return f"IF {' AND '.join(conditions)} THEN action = {action} (output_vector = {self.consequent}, support={self.support_count}{priority_str})"
        else:
            # 分类树输出：显示类别
            return f"IF {' AND '.join(conditions)} THEN class = {self.consequent} (support={self.support_count}{priority_str})"

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

    def get_best_action(self, mask: Optional[np.ndarray] = None) -> int:
        """
        获取最佳动作（考虑mask）

        Args:
            mask: 合法动作的mask（True表示合法），如果为None则不考虑mask

        Returns:
            最佳动作的索引
        """
        if self.output_vector is None:
            # 分类树：直接返回consequent
            if isinstance(self.consequent, np.ndarray):
                return int(np.argmax(self.consequent))
            return int(self.consequent)

        # 回归树：从output_vector中选择最佳动作
        logits = self.output_vector.copy()

        if mask is not None:
            # 应用mask：将不合法的动作设置为极小值
            logits[~mask] = -np.inf

        return int(np.argmax(logits))

    def to_dict(self) -> Dict:
        """将规则转换为字典格式"""
        # 转换antecedents，确保所有值都是Python原生类型
        antecedents_list = []
        for feature_idx, operator, value in self.antecedents:
            antecedents_list.append([
                int(feature_idx),
                str(operator),
                float(value)
            ])

        result = {
            'antecedents': antecedents_list,
            'support_count': int(self.support_count),
            'priority': float(self.priority),
            'rule_string': str(self)
        }

        # 处理consequent和output_vector
        if isinstance(self.consequent, np.ndarray):
            result['output_vector'] = [float(x) for x in self.consequent.tolist()]
            result['best_action'] = int(np.argmax(self.consequent))
        else:
            result['consequent'] = int(self.consequent)

        if self.output_vector is not None:
            result['output_vector'] = [float(x) for x in self.output_vector.tolist()]

        return result


class DecisionTreeRuleExtractor:
    """决策树规则提取和简化器"""

    def __init__(self, tree_model,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 feature_names: Optional[List[str]] = None,
                 alpha: float = 0.05,
                 oracle_model=None,
                 env=None):
        """
        初始化规则提取器

        Args:
            tree_model: 训练好的sklearn决策树模型（DecisionTreeClassifier或DecisionTreeRegressor）
            X_train: 训练数据特征
            y_train: 训练数据标签（对于回归树可以是多维向量）
            feature_names: 特征名称列表（可选）
            alpha: 统计检验显著性水平（默认0.05）
            oracle_model: Oracle模型（用于计算状态重要性，可选）
            env: 环境对象（用于计算状态重要性，可选）
        """
        self.tree = tree_model
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names
        self.alpha = alpha
        self.oracle_model = oracle_model
        self.env = env
        self.rules: List[Rule] = []
        self._extraction_stats = {}

        # 检测树的类型
        from sklearn.tree import DecisionTreeRegressor
        self.is_regressor = isinstance(tree_model, DecisionTreeRegressor)

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
                # 获取该叶节点的值
                values = tree_.value[node]
                print('values shape', values.shape, 'values', values)
                reshaped_values = values.reshape(-1)
                print('reshape_values shape', reshaped_values.shape, 'reshaped_values', reshaped_values)
                values = reshaped_values

                if self.is_regressor:
                    # 回归树：values是预测的向量 (1, n_outputs) 或 (n_outputs,)
                    if values.ndim == 2:
                        output_vector = values[0]  # shape: (n_outputs,)
                    else:
                        output_vector = values

                    # 支持计数从训练数据中计算
                    support_count = tree_.n_node_samples[node]

                    # consequent是完整的输出向量
                    rule = Rule(
                        antecedents.copy(),
                        consequent=output_vector.copy(),
                        support_count=support_count,
                        output_vector=output_vector.copy()
                    )
                    self.rules.append(rule)
                else:
                    # 分类树：values是类别计数 (1, n_classes)
                    values = values[0]
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

    def compute_rule_state(self, rule: Rule) -> Optional[np.ndarray]:
        """
        将规则的前件转换为对应的状态表示

        根据规则的前件条件构造一个满足条件的最简单状态。

        Args:
            rule: 规则对象

        Returns:
            状态向量（observation），如果无法构造则返回None
        """
        if len(rule.antecedents) == 0:
            # 没有前件（默认规则），返回None
            return None

        # 创建一个全0状态（空棋盘）
        state = np.zeros(self.tree.n_features_in_)

        # 根据规则的前件设置状态值
        # 对于井字棋：状态值为 0（空）、1（己方）、-1（对手）
        for feature_idx, operator, value in rule.antecedents:
            if operator == '<=':
                # 满足 X[i] <= value
                if value >= 0.5:
                    state[feature_idx] = 0  # 空位（0 <= 0.5）
                elif value >= -0.5:
                    state[feature_idx] = -1  # 对手（-1 <= 0）
                else:
                    state[feature_idx] = -1  # 对手
            else:  # operator == '>'
                # 满足 X[i] > value
                if value < -0.5:
                    state[feature_idx] = -1  # 对手（-1 > -1 不成立，这里用0）
                    state[feature_idx] = 0   # 实际上用0
                elif value < 0.5:
                    state[feature_idx] = 1   # 己方（1 > 0）
                else:
                    state[feature_idx] = 1   # 己方

        return state
    
    def compute_mask_from_state(self, state: np.ndarray) -> np.ndarray:
        """
        从状态向量计算动作mask

        Args:
            state: 状态向量

        Returns:
            mask: 布尔数组，True表示该动作合法
        """
        # 对于井字棋，空位置（0）为合法动作
        mask = (state == 0)
        return mask

    def compute_state_criticality_by_tree(self, rule: Rule, mask) -> float:
        """
        通过决策树的输出计算状态重要性
        mask: True表示合法动作(空位置)，False表示非法动作(已占据位置)
        """
        output_vector = rule.output_vector[mask]
        # 如果没有合法动作，返回0
        if len(output_vector) == 0:
            return 0.0
        criticality = np.max(output_vector) - np.min(output_vector)
        print('output_vector before masking shape', rule.output_vector.shape, 'output_vector before masking', rule.output_vector)
        print('mask shape', mask.shape, 'mask', mask)
        print('output_vector shape', output_vector.shape, 'output_vector', output_vector)
        print('criticality', criticality)
        return float(criticality)
        

    def compute_state_criticality(self, observation: np.ndarray) -> float:
        """
        计算给定状态的重要性（criticality）

        使用viper_mask_ppo中的compute_criticality函数的逻辑

        Args:
            observation: 状态向量

        Returns:
            状态重要性得分（float）
        """
        if self.oracle_model is None or self.env is None:
            return 0.0

        try:
            import torch

            # 确保observation是正确的形状
            if observation.ndim == 1:
                obs = observation
            else:
                obs = observation.flatten()

            # 计算mask（对于井字棋，空位置为True）
            mask = (obs == 0).astype(bool)
            mask_tensor = torch.tensor(mask).unsqueeze(0)

            # 获取所有可能的动作
            possible_actions = np.where(mask)[0]

            if len(possible_actions) == 0:
                return 0.0

            obs_tensor = torch.as_tensor(obs).unsqueeze(0).to(self.oracle_model.device)

            # 计算每个动作的log概率
            log_probs = []
            for action in possible_actions:
                action_tensor = torch.tensor([action]).to(self.oracle_model.device)
                _, log_prob, _ = self.oracle_model.policy.evaluate_actions(
                    obs_tensor, action_tensor, action_masks=mask_tensor
                )
                log_probs.append(log_prob.detach().cpu().numpy().flatten()[0])

            log_probs = np.array(log_probs)

            # 计算criticality: max(log_prob) - min(log_prob)
            criticality = log_probs.max() - log_probs.min()

            return float(criticality)

        except Exception as e:
            warnings.warn(f"计算状态重要性时出错: {e}")
            return 0.0
        
    def compute_rule_priorities_by_tree(self, verbose: bool = False) -> List[Rule]:
        """
        为所有规则计算优先级（基于状态重要性）

        Args:
            verbose: 是否打印详细信息

        Returns:
            更新了优先级的规则列表
        """
        if self.oracle_model is None or self.env is None:
            if verbose:
                print("警告: 未提供oracle模型或环境，无法计算优先级")
            return self.rules

        if verbose:
            print(f"\n计算 {len(self.rules)} 条规则的优先级...")

        for i, rule in enumerate(self.rules):
            # 找到代表该规则的状态
            state = self.compute_rule_state(rule)

            if state is not None:
                # 计算该状态的重要性
                # priority = self.compute_state_criticality(state)
                priority = self.compute_state_criticality_by_tree(rule, mask=self.compute_mask_from_state(state))
                rule.priority = priority

                if verbose and (i < 10 or i % 100 == 0):
                    print(f"  规则 {i+1}: priority={priority:.4f}")
            else:
                rule.priority = 0.0
                if verbose:
                    print(f"  规则 {i+1}: 无法找到匹配状态，priority=0.0")

        if verbose:
            print(f"完成优先级计算")
            priorities = [r.priority for r in self.rules]
            print(f"  优先级范围: [{min(priorities):.4f}, {max(priorities):.4f}]")
            print(f"  平均优先级: {np.mean(priorities):.4f}")

        return self.rules

    def compute_rule_priorities(self, verbose: bool = False) -> List[Rule]:
        """
        为所有规则计算优先级（基于状态重要性）

        Args:
            verbose: 是否打印详细信息

        Returns:
            更新了优先级的规则列表
        """
        if self.oracle_model is None or self.env is None:
            if verbose:
                print("警告: 未提供oracle模型或环境，无法计算优先级")
            return self.rules

        if verbose:
            print(f"\n计算 {len(self.rules)} 条规则的优先级...")

        for i, rule in enumerate(self.rules):
            # 找到代表该规则的状态
            state = self.compute_rule_state(rule)

            if state is not None:
                # 计算该状态的重要性
                priority = self.compute_state_criticality(state)
                rule.priority = priority

                if verbose and (i < 10 or i % 100 == 0):
                    print(f"  规则 {i+1}: priority={priority:.4f}")
            else:
                rule.priority = 0.0
                if verbose:
                    print(f"  规则 {i+1}: 无法找到匹配状态，priority=0.0")

        if verbose:
            print(f"完成优先级计算")
            priorities = [r.priority for r in self.rules]
            print(f"  优先级范围: [{min(priorities):.4f}, {max(priorities):.4f}]")
            print(f"  平均优先级: {np.mean(priorities):.4f}")

        return self.rules

    def sort_rules_by_priority(self, descending: bool = True) -> List[Rule]:
        """
        按优先级对规则进行排序

        Args:
            descending: True表示降序（优先级高的在前），False表示升序

        Returns:
            排序后的规则列表
        """
        self.rules = sorted(self.rules, key=lambda r: r.priority, reverse=descending)
        return self.rules

    def get_stats(self) -> Dict:
        """获取提取统计信息"""
        stats = self._extraction_stats.copy()

        # 添加优先级统计
        if self.rules:
            priorities = [r.priority for r in self.rules]
            stats['priority_min'] = float(np.min(priorities))
            stats['priority_max'] = float(np.max(priorities))
            stats['priority_mean'] = float(np.mean(priorities))
            stats['priority_std'] = float(np.std(priorities))

        return stats

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

    def export_rules_to_text(self, filepath: str, include_vectors: bool = True, metadata: dict = None):
        """
        将规则导出到文本文件

        Args:
            filepath: 输出文件路径
            include_vectors: 是否包含完整的输出向量（对于回归树）
            metadata: 元数据字典，包含提取配置信息
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"决策树规则提取结果\n")
            f.write(f"{'='*80}\n\n")

            # 写入元数据（如果提供）
            if metadata:
                from datetime import datetime
                f.write(f"提取配置:\n")
                f.write(f"  生成时间: {metadata.get('timestamp', 'N/A')}\n")
                f.write(f"  决策树路径: {metadata.get('tree_path', 'N/A')}\n")
                f.write(f"  决策树大小: {metadata.get('tree_size', 'N/A')} 个叶节点\n")
                if metadata.get('oracle_path'):
                    f.write(f"  Oracle路径: {metadata.get('oracle_path', 'N/A')}\n")
                f.write(f"  规则简化: {'否' if metadata.get('no_simplify', False) else '是'}\n")
                if metadata.get('compute_priority'):
                    priority_method = '决策树输出' if metadata.get('priority_by_tree', False) else 'Oracle输出'
                    f.write(f"  优先级计算: 是 (基于{priority_method})\n")
                else:
                    f.write(f"  优先级计算: 否\n")
                if metadata.get('sort_by_priority'):
                    f.write(f"  按优先级排序: 是\n")
                if not metadata.get('no_simplify', False):
                    f.write(f"  简化样本数: {metadata.get('n_samples', 'N/A')}\n")
                    f.write(f"  显著性水平α: {metadata.get('alpha', 'N/A')}\n")
                f.write(f"\n")

            # 写入树类型
            tree_type = "回归树 (Regression)" if self.is_regressor else "分类树 (Classification)"
            f.write(f"树类型: {tree_type}\n\n")

            f.write(f"统计信息:\n")
            for key, value in self._extraction_stats.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\n{'='*80}\n\n")

            f.write(f"规则列表 (共 {len(self.rules)} 条):\n\n")
            for i, rule in enumerate(self.rules, 1):
                # 写入规则条件
                if rule.antecedents:
                    conditions = []
                    for feature_idx, operator, value in rule.antecedents:
                        conditions.append(f"X[{feature_idx}] {operator} {value:.3f}")
                    f.write(f"规则 {i:3d}: IF {' AND '.join(conditions)}\n")
                else:
                    f.write(f"规则 {i:3d}: DEFAULT RULE\n")

                # 写入规则输出
                if rule.output_vector is not None and include_vectors:
                    # 回归树：显示完整的9维向量
                    best_action = np.argmax(rule.output_vector)
                    f.write(f"       THEN best_action = {best_action}\n")
                    f.write(f"       output_vector = {rule.output_vector}\n")
                    f.write(f"       (support={rule.support_count}")
                    if rule.priority > 0:
                        f.write(f", priority={rule.priority:.4f}")
                    f.write(f")\n")
                elif isinstance(rule.consequent, np.ndarray):
                    # 回归树但不显示向量
                    best_action = np.argmax(rule.consequent)
                    f.write(f"       THEN best_action = {best_action}\n")
                    f.write(f"       (support={rule.support_count}")
                    if rule.priority > 0:
                        f.write(f", priority={rule.priority:.4f}")
                    f.write(f")\n")
                else:
                    # 分类树
                    f.write(f"       THEN class = {rule.consequent}\n")
                    f.write(f"       (support={rule.support_count}")
                    if rule.priority > 0:
                        f.write(f", priority={rule.priority:.4f}")
                    f.write(f")\n")

                f.write("\n")

        print(f"规则已导出到: {filepath}")

    def export_rules_to_json(self, filepath: str):
        """
        将规则导出到JSON文件（便于程序读取）

        Args:
            filepath: 输出JSON文件路径
        """
        import json

        output = {
            'tree_type': 'regressor' if self.is_regressor else 'classifier',
            'statistics': self.get_stats(),
            'rules': [rule.to_dict() for rule in self.rules]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"规则已导出到JSON: {filepath}")


def extract_and_simplify_rules(tree_model,
                                X_train: np.ndarray,
                                y_train: np.ndarray,
                                feature_names: Optional[List[str]] = None,
                                alpha: float = 0.05,
                                oracle_model=None,
                                env=None,
                                compute_priority: bool = False,
                                sort_by_priority: bool = False,
                                verbose: bool = True) -> DecisionTreeRuleExtractor:
    """
    便捷函数：提取并简化决策树规则

    Args:
        tree_model: 训练好的sklearn决策树模型
        X_train: 训练数据特征
        y_train: 训练数据标签
        feature_names: 特征名称列表
        alpha: 显著性水平
        oracle_model: Oracle模型（用于计算状态重要性，可选）
        env: 环境对象（用于计算状态重要性，可选）
        compute_priority: 是否计算规则优先级
        sort_by_priority: 是否按优先级排序规则
        verbose: 是否打印详细信息

    Returns:
        DecisionTreeRuleExtractor实例
    """
    extractor = DecisionTreeRuleExtractor(tree_model, X_train, y_train,
                                         feature_names, alpha, oracle_model, env)

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

    # 计算优先级
    if compute_priority:
        if verbose:
            print("\n" + "="*80)
            print("步骤 4: 计算规则优先级")
            print("="*80)

        extractor.compute_rule_priorities(verbose=verbose)
        #extractor.compute_rule_priorities_by_tree(verbose=verbose)

        # 按优先级排序
        if sort_by_priority:
            if verbose:
                print("\n按优先级排序规则（降序）...")
            extractor.sort_rules_by_priority(descending=True)

            if verbose:
                print("\n优先级最高的10条规则:")
                extractor.print_rules(max_rules=10)

    return extractor
