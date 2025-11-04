"""
暴力枚举测试：测试 PPO 模型在所有 winning/losing 条件下的表现

该脚本会：
1. 枚举所有即将获胜的棋局（one move to win）
2. 通过反转棋盘生成对应的防守棋局（one move to defend）
3. 测试模型是否能在这些关键时刻做出正确决策
4. 统计并报告成功率

策略：
- 生成所有"X只需一步即可获胜"的棋局
- 将棋盘反转（X<->O）得到"X必须防守否则会输"的棋局
"""

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from typing import List, Tuple, Dict
import sys
import os
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ExhaustiveWinLoseTest:
    """暴力枚举所有 winning/losing 条件"""

    # 所有获胜组合
    WIN_COMBINATIONS = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # 行
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # 列
        [0, 4, 8], [2, 4, 6]              # 对角线
    ]

    def __init__(self, model_path: str):
        """
        Args:
            model_path: 训练好的 PPO 模型路径
        """
        self.model_path = model_path
        self.model = None

        # 加载模型
        self._load_model()

    def _load_model(self):
        """加载模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        print(f"加载模型: {self.model_path}")
        self.model = MaskablePPO.load(self.model_path)
        print("✓ 模型加载成功\n")

    def _predict_action(self, board: np.ndarray, deterministic: bool = True) -> int:
        """
        使用模型预测动作

        Args:
            board: 棋盘状态 (9,)
            deterministic: 是否使用确定性策略

        Returns:
            action: 预测的动作 (0-8)
        """
        # 创建 action mask
        mask = (board == 0).astype(np.int8)
        mask_tensor = torch.tensor(mask).unsqueeze(0).to(self.model.device)

        # 预测
        obs_tensor = torch.tensor(board).float().unsqueeze(0).to(self.model.device)

        with torch.no_grad():
            action, _ = self.model.predict(obs_tensor.cpu().numpy(),
                                          deterministic=deterministic,
                                          action_masks=mask_tensor.cpu().numpy())

        return int(action)

    def _visualize_board(self, board: np.ndarray, highlight_pos: int = None) -> str:
        """
        可视化棋盘

        Args:
            board: 棋盘状态
            highlight_pos: 高亮位置（用于显示正确答案）
        """
        symbols = {0: '.', 1: 'X', -1: 'O'}
        lines = []

        for i in range(0, 9, 3):
            row_symbols = []
            for j in range(i, i+3):
                if highlight_pos is not None and j == highlight_pos:
                    row_symbols.append(f"[{symbols[board[j]]}]")
                else:
                    row_symbols.append(f" {symbols[board[j]]} ")

            lines.append(f"  {'|'.join(row_symbols)}")
            if i < 6:
                lines.append(f"  -----------")

        return "\n".join(lines)

    def generate_critical_cases(self) -> List[Tuple[np.ndarray, int, str]]:
        """
        生成所有"一步获胜"的测试案例（关键棋局）

        策略：
        - 对于每个获胜组合，枚举所有可能的"X差一步获胜"的情况
        - 即：该组合中有2个X，1个空位
        - 同时在其余位置放置合理数量的O，保证棋局合法

        返回: [(棋盘状态, 正确动作, 描述), ...]
        """
        test_cases = []

        for win_combo in self.WIN_COMBINATIONS:
            # 对于每个获胜组合，生成所有可能的"差一步获胜"的情况
            # 即：该组合中有2个X，1个空位

            for empty_pos in win_combo:
                # 该组合中，empty_pos 是空的，其余两个是 X
                filled_positions = [pos for pos in win_combo if pos != empty_pos]

                # 现在我们需要在其余位置放置一些 O（对手棋子）
                # 使得棋局合法（X和O数量接近）

                remaining_positions = [i for i in range(9) if i not in win_combo]

                # 生成不同数量的 O 的组合（0到3个O）
                for num_o in range(0, min(4, len(remaining_positions) + 1)):
                    if num_o > len(remaining_positions):
                        continue

                    # 如果没有O，生成一个空棋盘的情况
                    if num_o == 0:
                        o_combinations = [tuple()]
                    else:
                        o_combinations = list(combinations(remaining_positions, num_o))

                    for o_positions in o_combinations:
                        # 构造棋盘
                        board = np.zeros(9, dtype=np.float32)

                        # 放置 X（获胜组合中的两个位置）
                        for pos in filled_positions:
                            board[pos] = 1

                        # 放置 O
                        for pos in o_positions:
                            board[pos] = -1

                        # 检查棋局合法性（X 和 O 数量差不超过1）
                        num_x = np.sum(board == 1)
                        num_o = np.sum(board == -1)

                        if abs(num_x - num_o) > 1:
                            continue

                        # X应该先手，所以X数量应该 >= O数量
                        if num_x < num_o:
                            continue

                        # 检查 O 是否已经形成获胜（这种情况下棋局已经结束）
                        if self._check_winner(board, -1):
                            continue

                        # 生成描述
                        combo_name = self._get_combo_name(win_combo)
                        desc = f"Win {combo_name}, fill pos {empty_pos}"

                        test_cases.append((board.copy(), empty_pos, desc))

        return test_cases

    def _flip_board(self, board: np.ndarray) -> np.ndarray:
        """
        反转棋盘：将 X 和 O 互换

        Args:
            board: 原始棋盘

        Returns:
            flipped_board: 反转后的棋盘
        """
        return -board

    def _check_winner(self, board: np.ndarray, player: int) -> bool:
        """检查某个玩家是否已经获胜"""
        for combo in self.WIN_COMBINATIONS:
            if all(board[pos] == player for pos in combo):
                return True
        return False

    def _get_combo_name(self, combo: List[int]) -> str:
        """获取获胜组合的名称"""
        if combo == [0, 1, 2]:
            return "Row0"
        elif combo == [3, 4, 5]:
            return "Row1"
        elif combo == [6, 7, 8]:
            return "Row2"
        elif combo == [0, 3, 6]:
            return "Col0"
        elif combo == [1, 4, 7]:
            return "Col1"
        elif combo == [2, 5, 8]:
            return "Col2"
        elif combo == [0, 4, 8]:
            return "Diag_main"
        elif combo == [2, 4, 6]:
            return "Diag_anti"
        else:
            return "Unknown"

    def test_winning_cases(self, test_cases: List[Tuple[np.ndarray, int, str]],
                          verbose: bool = False, max_display: int = 5) -> Dict:
        """
        测试所有"一步获胜"的情况

        Args:
            test_cases: 测试案例列表
            verbose: 是否显示详细信息
            max_display: 最多显示多少个失败案例

        Returns:
            结果字典
        """
        print("=" * 70)
        print("测试 1: 一步获胜（Winning Cases）")
        print("=" * 70)
        print(f"测试案例数: {len(test_cases)}\n")

        print("开始测试模型...")
        correct = 0
        failed_cases = []

        for i, (board, correct_action, desc) in enumerate(test_cases):
            predicted_action = self._predict_action(board, deterministic=True)

            is_correct = (predicted_action == correct_action)

            if is_correct:
                correct += 1
            else:
                failed_cases.append((board, correct_action, predicted_action, desc))

            if verbose and not is_correct:
                print(f"\n✗ [{i+1}/{len(test_cases)}] {desc}")
                print(self._visualize_board(board, highlight_pos=correct_action))
                print(f"  正确: {correct_action}, 预测: {predicted_action}")

        accuracy = correct / len(test_cases) * 100

        print(f"\n结果:")
        print(f"  正确: {correct}/{len(test_cases)} ({accuracy:.2f}%)")
        print(f"  错误: {len(test_cases) - correct}")

        # 显示部分失败案例
        if len(failed_cases) > 0 and not verbose:
            print(f"\n失败案例示例 (显示前{min(max_display, len(failed_cases))}个):")
            for i, (board, correct_action, predicted_action, desc) in enumerate(failed_cases[:max_display]):
                print(f"\n案例 {i+1}: {desc}")
                print(self._visualize_board(board, highlight_pos=correct_action))
                print(f"  正确动作: {correct_action}")
                print(f"  预测动作: {predicted_action}")

        # 评级
        if accuracy >= 95:
            grade = "优秀 ✓✓"
        elif accuracy >= 85:
            grade = "良好 ✓"
        elif accuracy >= 70:
            grade = "及格"
        else:
            grade = "不及格 ✗"

        print(f"\n评级: {grade}")

        return {
            'total': len(test_cases),
            'correct': correct,
            'accuracy': accuracy,
            'failed_cases': failed_cases,
            'grade': grade
        }

    def test_losing_cases(self, test_cases: List[Tuple[np.ndarray, int, str]],
                         verbose: bool = False, max_display: int = 5) -> Dict:
        """
        测试所有"一步防守"的情况
        通过反转原始测试案例（X<->O）生成防守案例

        Args:
            test_cases: 原始测试案例列表
            verbose: 是否显示详细信息
            max_display: 最多显示多少个失败案例

        Returns:
            结果字典
        """
        print("\n" + "=" * 70)
        print("测试 2: 一步防守（Losing Cases - Must Block）")
        print("=" * 70)
        print(f"测试案例数: {len(test_cases)} (通过反转获胜案例生成)\n")

        print("开始测试模型...")
        correct = 0
        failed_cases = []

        for i, (board, correct_action, desc) in enumerate(test_cases):
            # 反转棋盘：X<->O
            flipped_board = self._flip_board(board)

            # 动作位置保持不变，但描述改为"防守"
            defend_desc = desc.replace("Win", "Block")

            predicted_action = self._predict_action(flipped_board, deterministic=True)

            is_correct = (predicted_action == correct_action)

            if is_correct:
                correct += 1
            else:
                failed_cases.append((flipped_board, correct_action, predicted_action, defend_desc))

            if verbose and not is_correct:
                print(f"\n✗ [{i+1}/{len(test_cases)}] {defend_desc}")
                print(self._visualize_board(flipped_board, highlight_pos=correct_action))
                print(f"  正确: {correct_action}, 预测: {predicted_action}")

        accuracy = correct / len(test_cases) * 100

        print(f"\n结果:")
        print(f"  正确: {correct}/{len(test_cases)} ({accuracy:.2f}%)")
        print(f"  错误: {len(test_cases) - correct}")

        # 显示部分失败案例
        if len(failed_cases) > 0 and not verbose:
            print(f"\n失败案例示例 (显示前{min(max_display, len(failed_cases))}个):")
            for i, (board, correct_action, predicted_action, desc) in enumerate(failed_cases[:max_display]):
                print(f"\n案例 {i+1}: {desc}")
                print(self._visualize_board(board, highlight_pos=correct_action))
                print(f"  正确动作: {correct_action}")
                print(f"  预测动作: {predicted_action}")

        # 评级
        if accuracy >= 95:
            grade = "优秀 ✓✓"
        elif accuracy >= 85:
            grade = "良好 ✓"
        elif accuracy >= 70:
            grade = "及格"
        else:
            grade = "不及格 ✗"

        print(f"\n评级: {grade}")

        return {
            'total': len(test_cases),
            'correct': correct,
            'accuracy': accuracy,
            'failed_cases': failed_cases,
            'grade': grade
        }

    def analyze_error_patterns(self, win_results: Dict, lose_results: Dict):
        """分析错误模式"""
        print("\n" + "=" * 70)
        print("错误模式分析")
        print("=" * 70)

        # 分析获胜案例的错误模式
        print("\n1. 获胜案例错误分布:")
        win_errors = defaultdict(int)
        for board, correct_action, predicted_action, desc in win_results['failed_cases']:
            combo_name = desc.split(',')[0].replace('Win ', '')
            win_errors[combo_name] += 1

        if len(win_errors) > 0:
            for combo, count in sorted(win_errors.items(), key=lambda x: x[1], reverse=True):
                print(f"  {combo}: {count} 次错误")
        else:
            print("  无错误 ✓")

        # 分析防守案例的错误模式
        print("\n2. 防守案例错误分布:")
        lose_errors = defaultdict(int)
        for board, correct_action, predicted_action, desc in lose_results['failed_cases']:
            combo_name = desc.split(',')[0].replace('Block ', '')
            lose_errors[combo_name] += 1

        if len(lose_errors) > 0:
            for combo, count in sorted(lose_errors.items(), key=lambda x: x[1], reverse=True):
                print(f"  {combo}: {count} 次错误")
        else:
            print("  无错误 ✓")

    def run_full_test(self, verbose: bool = False, max_display: int = 5):
        """运行完整的枚举测试"""
        print("\n" + "=" * 70)
        print("PPO 模型 - 暴力枚举 Winning/Losing 条件测试")
        print("=" * 70)
        print(f"模型: {self.model_path}\n")

        # 生成基础测试案例（获胜案例）
        print("生成测试案例...")
        test_cases = self.generate_critical_cases()
        print(f"✓ 生成了 {len(test_cases)} 个基础案例")
        print(f"✓ 将通过反转生成对应的防守案例\n")

        # 测试1: 获胜案例
        win_results = self.test_winning_cases(test_cases, verbose=verbose, max_display=max_display)

        # 测试2: 防守案例（通过反转获胜案例生成）
        lose_results = self.test_losing_cases(test_cases, verbose=verbose, max_display=max_display)

        # 错误模式分析
        self.analyze_error_patterns(win_results, lose_results)

        # 综合评估
        print("\n" + "=" * 70)
        print("综合评估")
        print("=" * 70)

        total_cases = win_results['total'] + lose_results['total']
        total_correct = win_results['correct'] + lose_results['correct']
        total_accuracy = total_correct / total_cases * 100

        print(f"\n总测试案例: {total_cases}")
        print(f"  - 获胜案例: {win_results['total']}")
        print(f"  - 防守案例: {lose_results['total']}")

        print(f"\n总正确率: {total_correct}/{total_cases} ({total_accuracy:.2f}%)")
        print(f"  - 获胜案例准确率: {win_results['accuracy']:.2f}%")
        print(f"  - 防守案例准确率: {lose_results['accuracy']:.2f}%")

        # 总体评级
        if total_accuracy >= 95:
            overall_grade = "优秀 ✓✓ - 模型在关键决策上表现出色！"
        elif total_accuracy >= 85:
            overall_grade = "良好 ✓ - 模型在关键决策上表现良好"
        elif total_accuracy >= 70:
            overall_grade = "及格 - 模型基本掌握关键决策"
        else:
            overall_grade = "不及格 ✗ - 模型在关键决策上需要改进"

        print(f"\n总体评级: {overall_grade}")
        print("=" * 70 + "\n")

        return {
            'win_results': win_results,
            'lose_results': lose_results,
            'total_cases': total_cases,
            'total_correct': total_correct,
            'total_accuracy': total_accuracy,
            'overall_grade': overall_grade
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='暴力枚举测试 PPO 模型的 Winning/Losing 决策能力')
    parser.add_argument('--model', type=str,
                       default='log/oracle_TicTacToe_ppo_aggressive.zip',
                       help='模型路径')
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细测试过程（包括所有失败案例）')
    parser.add_argument('--max-display', type=int, default=10,
                       help='最多显示多少个失败案例（当不使用verbose时）')

    args = parser.parse_args()

    # 创建测试器
    tester = ExhaustiveWinLoseTest(args.model)

    # 运行完整测试
    results = tester.run_full_test(verbose=args.verbose, max_display=args.max_display)

    print("\n测试完成！")


if __name__ == "__main__":
    main()
