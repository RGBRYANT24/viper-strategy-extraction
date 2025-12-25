#!/usr/bin/env python3
"""
决策树与导出规则一致性检测

测试在所有可能状态下，决策树和导出的规则文件是否有相同的输出。
这个工具用于发现精度丢失和逻辑错误。

使用示例:
    # 基本测试
    python test/test_tree_rule_consistency.py \
        --tree log/viper_X-only/tree.joblib \
        --rules log/viper_X-only/rules.json \
        --n-samples 1000

    # 详细模式（显示所有不匹配）
    python test/test_tree_rule_consistency.py \
        --tree log/viper_X-only/tree.joblib \
        --rules log/viper_X-only/rules.json \
        --n-samples 1000 \
        --verbose \
        --max-mismatches 10

    # 穷举所有可能状态（需要较长时间）
    python test/test_tree_rule_consistency.py \
        --tree log/viper_X-only/tree.joblib \
        --rules log/viper_X-only/rules.json \
        --exhaustive
"""

import json
import joblib
import numpy as np
import gymnasium as gym
import sys
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# 确保能导入您的环境
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入环境
try:
    import gym_env
except ImportError:
    pass


class TreeRuleConsistencyChecker:
    """决策树与规则一致性检测器"""

    def __init__(self, tree_path: str, rules_path: str, verbose: bool = False):
        """
        初始化检测器

        Args:
            tree_path: 决策树模型路径（.joblib）
            rules_path: 规则文件路径（.json）
            verbose: 是否显示详细信息
        """
        self.verbose = verbose

        # 加载决策树
        print(f"正在加载决策树: {tree_path}")
        self.tree = joblib.load(tree_path)

        # 加载规则
        print(f"正在加载规则: {rules_path}")
        with open(rules_path, 'r') as f:
            rules_data = json.load(f)

        self.rules_list = rules_data['rules']
        self.tree_type = rules_data.get('tree_type', 'regressor')

        print(f"加载成功: {len(self.rules_list)} 条规则")
        print(f"树类型: {self.tree_type}")

        # 统计信息
        self.stats = {
            'total_samples': 0,
            'vector_mismatches': 0,
            'argmax_mismatches': 0,
            'no_rule_match': 0,
            'max_value_diff': 0.0,
            'avg_value_diff': 0.0,
            'value_diffs': []
        }

        # 不匹配详情（用于调试）
        self.mismatch_details = []

    def find_matching_rule(self, obs: np.ndarray) -> Optional[Dict]:
        """
        找到匹配当前状态的规则

        Args:
            obs: 状态向量

        Returns:
            匹配的规则字典，如果没有匹配则返回None
        """
        for rule in self.rules_list:
            match = True
            for feat_idx, op, val in rule['antecedents']:
                feat_val = obs[feat_idx]
                if op == '<=':
                    if not (feat_val <= val):
                        match = False
                        break
                elif op == '>':
                    if not (feat_val > val):
                        match = False
                        break

            if match:
                return rule

        return None

    def get_tree_output(self, obs: np.ndarray) -> np.ndarray:
        """
        获取决策树的原始输出向量

        Args:
            obs: 状态向量

        Returns:
            输出向量
        """
        obs_reshaped = obs.reshape(1, -1)

        if hasattr(self.tree, 'predict_proba'):
            # 分类树
            tree_out = self.tree.predict_proba(obs_reshaped)[0]
        else:
            # 回归树
            tree_out = self.tree.predict(obs_reshaped)[0]

        return tree_out

    def compare_outputs(self, obs: np.ndarray,
                       apply_mask: bool = True) -> Tuple[bool, Dict]:
        """
        比较决策树和规则在给定状态下的输出

        Args:
            obs: 状态向量
            apply_mask: 是否应用合法动作mask

        Returns:
            (is_consistent, detail_dict)
        """
        # 获取树的输出
        tree_out = self.get_tree_output(obs)

        # 找到匹配的规则
        matched_rule = self.find_matching_rule(obs)

        if matched_rule is None:
            self.stats['no_rule_match'] += 1
            return False, {
                'error': 'no_rule_match',
                'state': obs.copy(),
                'tree_output': tree_out.copy()
            }

        # 获取规则的输出
        if 'output_vector' in matched_rule:
            rule_out = np.array(matched_rule['output_vector'])
        else:
            # 如果规则没有output_vector（可能是分类树），创建one-hot向量
            rule_out = np.zeros(len(tree_out))
            best_action = matched_rule.get('best_action', matched_rule.get('consequent', 0))
            rule_out[best_action] = 1.0

        # 计算向量差异
        diff = np.abs(tree_out - rule_out)
        max_diff = np.max(diff)
        avg_diff = np.mean(diff)

        self.stats['value_diffs'].append(max_diff)
        self.stats['max_value_diff'] = max(self.stats['max_value_diff'], max_diff)

        # 检查向量是否有显著差异（考虑浮点精度）
        EPSILON = 1e-9
        vector_match = np.allclose(tree_out, rule_out, atol=EPSILON, rtol=1e-5)

        if not vector_match:
            self.stats['vector_mismatches'] += 1

        # 计算argmax（考虑mask）
        if apply_mask:
            mask = (obs == 0).astype(bool)
        else:
            mask = np.ones(len(tree_out), dtype=bool)

        # 应用mask到输出向量
        tree_out_masked = tree_out.copy()
        rule_out_masked = rule_out.copy()
        tree_out_masked[~mask] = -np.inf
        rule_out_masked[~mask] = -np.inf

        tree_action = np.argmax(tree_out_masked)
        rule_action = np.argmax(rule_out_masked)

        argmax_match = (tree_action == rule_action)

        if not argmax_match:
            self.stats['argmax_mismatches'] += 1

        # 详细信息
        detail = {
            'state': obs.copy(),
            'mask': mask.copy(),
            'tree_output': tree_out.copy(),
            'rule_output': rule_out.copy(),
            'tree_action': int(tree_action),
            'rule_action': int(rule_action),
            'max_diff': float(max_diff),
            'avg_diff': float(avg_diff),
            'vector_match': vector_match,
            'argmax_match': argmax_match,
            'rule_id': matched_rule.get('rule_id', -1),
            'rule_priority': matched_rule.get('priority', 0.0)
        }

        is_consistent = vector_match and argmax_match

        if not is_consistent:
            self.mismatch_details.append(detail)

        return is_consistent, detail

    def test_random_states(self, n_samples: int = 1000,
                          apply_mask: bool = True,
                          play_as_o_prob: float = 0.5) -> Dict:
        """
        测试随机采样的状态

        Args:
            n_samples: 采样数量
            apply_mask: 是否应用合法动作mask
            play_as_o_prob: 作为后手(O)的概率

        Returns:
            测试统计信息
        """
        print(f"\n正在测试 {n_samples} 个随机状态...")
        print(f"是否应用mask: {apply_mask}")
        print(f"先后手设置: play_as_o_prob={play_as_o_prob}")
        print("="*80)

        env = gym.make('TicTacToe-v0', play_as_o_prob=play_as_o_prob)

        for sample_idx in range(n_samples):
            obs, _ = env.reset()
            done = False
            step = 0

            while not done and step < 9:  # 井字棋最多9步
                self.stats['total_samples'] += 1

                # 比较输出
                is_consistent, detail = self.compare_outputs(obs, apply_mask)

                # 打印进度
                if self.verbose and not is_consistent:
                    print(f"\n[不一致 #{len(self.mismatch_details)}] 样本 {sample_idx}, 步骤 {step}")
                    self._print_detail(detail)

                # 随机走一步继续
                mask = (obs == 0).astype(bool)
                legal_actions = np.where(mask)[0]
                if len(legal_actions) == 0:
                    break

                action = np.random.choice(legal_actions)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                step += 1

            # 打印进度
            if (sample_idx + 1) % 100 == 0:
                print(f"已测试 {sample_idx + 1}/{n_samples} 个episode...")

        return self._compute_stats()

    def test_exhaustive_states(self, apply_mask: bool = True) -> Dict:
        """
        穷举测试所有可能的状态（警告：可能需要很长时间）

        Args:
            apply_mask: 是否应用合法动作mask

        Returns:
            测试统计信息
        """
        print("\n警告: 穷举测试可能需要很长时间...")
        print("正在测试所有可能的井字棋状态...")
        print("="*80)

        # 对于井字棋，状态空间是3^9 = 19683种可能
        # 但很多是不合法的（比如同时有5个X和1个O）
        # 我们通过游戏树递归生成所有合法状态

        def generate_all_states(board: np.ndarray, player: int, depth: int = 0):
            """递归生成所有合法状态"""
            # 检查是否已经结束
            winner = self._check_winner(board)
            if winner != 0 or depth >= 9:
                return

            # 测试当前状态
            self.stats['total_samples'] += 1
            is_consistent, detail = self.compare_outputs(board, apply_mask)

            if self.verbose and not is_consistent:
                print(f"\n[不一致 #{len(self.mismatch_details)}] 深度 {depth}")
                self._print_detail(detail)

            # 生成所有可能的下一步
            legal_moves = np.where(board == 0)[0]
            for move in legal_moves:
                new_board = board.copy()
                new_board[move] = player
                generate_all_states(new_board, -player, depth + 1)

            # 打印进度
            if self.stats['total_samples'] % 1000 == 0:
                print(f"已测试 {self.stats['total_samples']} 个状态...")

        # 从空棋盘开始
        empty_board = np.zeros(9)
        generate_all_states(empty_board, 1, 0)

        return self._compute_stats()

    def _check_winner(self, board: np.ndarray) -> int:
        """
        检查游戏是否结束
        返回: 1=X赢, -1=O赢, 0=未结束或平局
        """
        win_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # 行
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # 列
            [0, 4, 8], [2, 4, 6]              # 对角线
        ]

        for combo in win_combinations:
            if all(board[pos] == 1 for pos in combo):
                return 1
            elif all(board[pos] == -1 for pos in combo):
                return -1

        return 0

    def _compute_stats(self) -> Dict:
        """计算并返回统计信息"""
        total = self.stats['total_samples']

        if total == 0:
            return self.stats

        # 计算平均差异
        if len(self.stats['value_diffs']) > 0:
            self.stats['avg_value_diff'] = float(np.mean(self.stats['value_diffs']))

        # 计算百分比
        self.stats['vector_mismatch_rate'] = self.stats['vector_mismatches'] / total
        self.stats['argmax_mismatch_rate'] = self.stats['argmax_mismatches'] / total
        self.stats['no_rule_match_rate'] = self.stats['no_rule_match'] / total

        return self.stats

    def _print_detail(self, detail: Dict):
        """打印不一致的详细信息"""
        print(f"  状态: {detail['state']}")
        print(f"  Mask: {detail['mask']}")

        if 'tree_output' in detail:
            tree_top3 = detail['tree_output'][np.argsort(-detail['tree_output'])[:3]]
            rule_top3 = detail['rule_output'][np.argsort(-detail['rule_output'])[:3]]
            print(f"  Tree输出 (top 3): {tree_top3}")
            print(f"  Rule输出 (top 3): {rule_top3}")

        if 'tree_action' in detail:
            print(f"  Tree动作: {detail['tree_action']}")
            print(f"  Rule动作: {detail['rule_action']}")

        if 'max_diff' in detail:
            print(f"  最大数值差异: {detail['max_diff']:.20f}")
            print(f"  平均数值差异: {detail['avg_diff']:.20f}")

        if 'rule_id' in detail:
            print(f"  规则ID: {detail['rule_id']}")
            print(f"  规则优先级: {detail['rule_priority']:.4f}")

        print(f"  向量匹配: {detail.get('vector_match', 'N/A')}")
        print(f"  Argmax匹配: {detail.get('argmax_match', 'N/A')}")

    def print_report(self, max_mismatches: int = 5):
        """
        打印测试报告

        Args:
            max_mismatches: 最多显示多少个不一致案例
        """
        print("\n" + "="*80)
        print("一致性检测报告")
        print("="*80)

        stats = self.stats
        total = stats['total_samples']

        print(f"\n总测试样本数: {total}")
        print(f"无匹配规则: {stats['no_rule_match']} ({stats.get('no_rule_match_rate', 0):.2%})")
        print(f"向量数值不匹配: {stats['vector_mismatches']} ({stats.get('vector_mismatch_rate', 0):.2%})")
        print(f"Argmax不一致: {stats['argmax_mismatches']} ({stats.get('argmax_mismatch_rate', 0):.2%})")

        print(f"\n数值精度:")
        print(f"  平均数值误差: {stats['avg_value_diff']:.20f}")
        print(f"  最大数值误差: {stats['max_value_diff']:.20f}")

        # 打印不一致案例
        if len(self.mismatch_details) > 0:
            print(f"\n" + "="*80)
            print(f"不一致案例详情 (显示前 {min(max_mismatches, len(self.mismatch_details))} 个)")
            print("="*80)

            for i, detail in enumerate(self.mismatch_details[:max_mismatches], 1):
                print(f"\n案例 {i}:")
                self._print_detail(detail)

            if len(self.mismatch_details) > max_mismatches:
                print(f"\n... 还有 {len(self.mismatch_details) - max_mismatches} 个不一致案例未显示")

        # 结论
        print("\n" + "="*80)
        print("结论")
        print("="*80)

        if stats['argmax_mismatches'] > 0:
            print("❌ 发现动作选择不一致！")
            print("   可能原因：")
            print("   1. 规则导出时浮点数精度丢失")
            print("   2. 规则匹配逻辑有误")
            print("   3. Mask应用逻辑不同")
            print("\n   建议：")
            print("   - 在导出规则时使用更高精度（如hex格式）存储浮点数")
            print("   - 检查规则匹配的条件判断是否正确")
            print("   - 确保mask计算逻辑一致")
        elif stats['vector_mismatches'] > 0:
            print("⚠️  输出向量有细微差异，但动作选择一致")
            print("   这可能是浮点精度导致的，通常可以接受")
        elif stats['no_rule_match'] > 0:
            print("❌ 有状态无法匹配到规则！")
            print("   建议检查规则提取是否完整，是否覆盖了所有决策树路径")
        else:
            print("✅ 所有测试通过！决策树和规则输出完全一致")


def main():
    parser = argparse.ArgumentParser(
        description="测试决策树与导出规则的一致性",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本测试
  python test/test_tree_rule_consistency.py \\
      --tree log/viper_X-only/tree.joblib \\
      --rules log/viper_X-only/rules.json \\
      --n-samples 1000

  # 详细模式（显示所有不匹配）
  python test/test_tree_rule_consistency.py \\
      --tree log/viper_X-only/tree.joblib \\
      --rules log/viper_X-only/rules.json \\
      --n-samples 1000 \\
      --verbose \\
      --max-mismatches 10

  # 穷举所有可能状态
  python test/test_tree_rule_consistency.py \\
      --tree log/viper_X-only/tree.joblib \\
      --rules log/viper_X-only/rules.json \\
      --exhaustive
        """
    )

    # 必需参数
    parser.add_argument("--tree", type=str, required=True,
                       help="决策树模型路径（.joblib）")
    parser.add_argument("--rules", type=str, required=True,
                       help="规则文件路径（.json）")

    # 测试模式
    parser.add_argument("--n-samples", type=int, default=1000,
                       help="随机采样测试的样本数（默认1000）")
    parser.add_argument("--exhaustive", action='store_true',
                       help="穷举测试所有可能状态（警告：需要较长时间）")

    # 测试选项
    parser.add_argument("--no-mask", action='store_true',
                       help="不应用合法动作mask（测试原始输出）")
    parser.add_argument("--play-as-o-prob", type=float, default=0.5,
                       help="作为后手(O)的概率: 0.0=总是先手X, 0.5=随机(默认), 1.0=总是后手O")

    # 输出选项
    parser.add_argument("--verbose", action='store_true',
                       help="显示所有不一致的详细信息")
    parser.add_argument("--max-mismatches", type=int, default=5,
                       help="报告中最多显示多少个不一致案例（默认5）")

    # 其他参数
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")

    args = parser.parse_args()

    # 检查文件是否存在
    tree_path = Path(args.tree)
    rules_path = Path(args.rules)

    if not tree_path.exists():
        print(f"错误: 决策树文件不存在: {args.tree}")
        sys.exit(1)

    if not rules_path.exists():
        print(f"错误: 规则文件不存在: {args.rules}")
        sys.exit(1)

    # 设置随机种子
    np.random.seed(args.seed)

    print("="*80)
    print("决策树与规则一致性检测")
    print("="*80)
    print(f"决策树: {args.tree}")
    print(f"规则文件: {args.rules}")
    print("="*80)

    # 创建检测器
    checker = TreeRuleConsistencyChecker(
        args.tree,
        args.rules,
        verbose=args.verbose
    )

    # 运行测试
    apply_mask = not args.no_mask

    if args.exhaustive:
        checker.test_exhaustive_states(apply_mask=apply_mask)
    else:
        checker.test_random_states(n_samples=args.n_samples, apply_mask=apply_mask,
                                  play_as_o_prob=args.play_as_o_prob)

    # 打印报告
    checker.print_report(max_mismatches=args.max_mismatches)

    # 返回退出码
    if checker.stats['argmax_mismatches'] > 0 or checker.stats['no_rule_match'] > 0:
        sys.exit(1)  # 有不一致，返回错误码
    else:
        sys.exit(0)  # 所有测试通过


if __name__ == "__main__":
    main()
