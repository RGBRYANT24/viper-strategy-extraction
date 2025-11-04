"""
暴力枚举测试：测试 PPO 模型在所有 winning/losing 条件下的表现（修复版）

修复思路：
1. 枚举所有 3^9 = 19683 种棋盘状态
2. 筛选出合法且关键的局面（即将获胜/需要防守）
3. 正确处理先后手：
   - 神经网络训练时总是从当前玩家视角（己方=1，对手=-1）
   - 如果枚举到的是后手局面，需要反转棋盘再输入给神经网络
4. 确保棋局合法性：
   - 先手（现在轮到下）：己方棋子数 = 对手棋子数
   - 后手（现在轮到下）：己方棋子数 = 对手棋子数 - 1
"""

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from typing import List, Tuple, Dict
import sys
import os
from collections import defaultdict
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ExhaustiveWinLoseTest:
    """暴力枚举所有 winning/losing 条件"""

    WIN_COMBINATIONS = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # 行
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # 列
        [0, 4, 8], [2, 4, 6]              # 对角线
    ]

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        print(f"加载模型: {self.model_path}")
        self.model = MaskablePPO.load(self.model_path)
        print("✓ 模型加载成功\n")

    def _predict_action(self, board: np.ndarray, deterministic: bool = True) -> int:
        """使用模型预测动作"""
        mask = (board == 0).astype(np.int8)
        mask_tensor = torch.tensor(mask).unsqueeze(0).to(self.model.device)
        obs_tensor = torch.tensor(board).float().unsqueeze(0).to(self.model.device)

        with torch.no_grad():
            action, _ = self.model.predict(obs_tensor.cpu().numpy(),
                                          deterministic=deterministic,
                                          action_masks=mask_tensor.cpu().numpy())
        return int(action)

    def _visualize_board(self, board: np.ndarray, highlight_pos: int = None) -> str:
        """可视化棋盘"""
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

    def _check_winner(self, board: np.ndarray, player: int) -> bool:
        """检查某个玩家是否已经获胜"""
        for combo in self.WIN_COMBINATIONS:
            if all(board[pos] == player for pos in combo):
                return True
        return False

    def _get_combo_name(self, combo: List[int]) -> str:
        """获取获胜组合的名称"""
        combo_map = {
            tuple([0, 1, 2]): "Row0", tuple([3, 4, 5]): "Row1", tuple([6, 7, 8]): "Row2",
            tuple([0, 3, 6]): "Col0", tuple([1, 4, 7]): "Col1", tuple([2, 5, 8]): "Col2",
            tuple([0, 4, 8]): "Diag_main", tuple([2, 4, 6]): "Diag_anti"
        }
        return combo_map.get(tuple(combo), "Unknown")

    def _is_one_move_to_win(self, board: np.ndarray, player: int) -> List[Tuple[int, List[int]]]:
        """
        检查某个玩家是否差一步获胜

        Returns:
            [(获胜位置, 获胜组合), ...]
        """
        results = []
        for combo in self.WIN_COMBINATIONS:
            player_count = sum(1 for pos in combo if board[pos] == player)
            empty_count = sum(1 for pos in combo if board[pos] == 0)

            if player_count == 2 and empty_count == 1:
                empty_pos = [pos for pos in combo if board[pos] == 0][0]
                results.append((empty_pos, combo))

        return results

    def generate_all_test_cases(self):
        """
        枚举所有 3^9 种棋盘，筛选出合法的关键场景

        返回：
        - winning_cases: 获胜案例
        - defending_cases: 防守案例
        """
        print("枚举所有可能的棋盘状态 (3^9 = 19683)...")

        winning_cases = []
        defending_cases = []

        # 枚举所有棋盘
        for config in product([-1, 0, 1], repeat=9):
            board = np.array(config, dtype=np.float32)

            num_x = int(np.sum(board == 1))
            num_o = int(np.sum(board == -1))

            # 基本合法性检查
            # X是传统意义上的先手，所以 X数量 应该在 [O-1, O+1] 范围内
            if num_x < num_o - 1 or num_x > num_o + 1:
                continue

            # 检查是否已经有人获胜（游戏应该结束）
            if self._check_winner(board, 1) or self._check_winner(board, -1):
                continue

            # 判断当前应该谁下棋
            # 如果 num_x = num_o，说明X刚下完，现在轮到O
            # 如果 num_x = num_o + 1，说明O刚下完，现在轮到X
            # 如果 num_x = num_o - 1，说明X刚下完，现在轮到O（X是后手的情况）

            # 我们需要找"现在轮到X下"的局面
            x_to_move = (num_x == num_o) or (num_x == num_o - 1)

            if not x_to_move:
                continue

            # 从当前棋盘视角（X准备下）判断：
            # - X是第一手玩家吗？-> num_x == num_o（双方下了相同次数）
            # - X是第二手玩家吗？-> num_x == num_o - 1（O比X多下了一次）
            x_is_first_player = (num_x == num_o)

            # 检查X是否即将获胜
            x_winning_moves = self._is_one_move_to_win(board, 1)

            # 检查O是否即将获胜（X需要防守）
            o_winning_moves = self._is_one_move_to_win(board, -1)

            # 收集获胜案例
            for pos, combo in x_winning_moves:
                combo_name = self._get_combo_name(combo)
                turn_str = "先手" if x_is_first_player else "后手"
                desc = f"Win {combo_name}, pos {pos} ({turn_str})"

                # X=1代表当前玩家（己方），O=-1代表对手
                # 无论先手后手，这个视角都是正确的，不需要翻转
                winning_cases.append((board.copy(), pos, desc))

            # 收集防守案例
            for pos, combo in o_winning_moves:
                combo_name = self._get_combo_name(combo)
                turn_str = "先手" if x_is_first_player else "后手"
                desc = f"Block {combo_name}, pos {pos} ({turn_str})"

                # X=1代表当前玩家（己方），O=-1代表对手
                # 无论先手后手，这个视角都是正确的，不需要翻转
                defending_cases.append((board.copy(), pos, desc))

        print(f"✓ 找到 {len(winning_cases)} 个获胜案例")
        print(f"✓ 找到 {len(defending_cases)} 个防守案例\n")

        return winning_cases, defending_cases

    def test_cases(self, test_cases: List[Tuple[np.ndarray, int, str]],
                   title: str, verbose: bool = False, max_display: int = 5) -> Dict:
        """测试案例"""
        print("=" * 70)
        print(title)
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

        accuracy = correct / len(test_cases) * 100 if len(test_cases) > 0 else 0

        print(f"\n结果:")
        print(f"  正确: {correct}/{len(test_cases)} ({accuracy:.2f}%)")
        print(f"  错误: {len(test_cases) - correct}")

        # 显示部分失败案例
        if len(failed_cases) > 0 and not verbose:
            print(f"\n失败案例示例 (显示前{min(max_display, len(failed_cases))}个):")
            for i, (board, correct_action, predicted_action, desc) in enumerate(failed_cases[:max_display]):
                num_x = int(np.sum(board == 1))
                num_o = int(np.sum(board == -1))
                print(f"\n案例 {i+1}: {desc}")
                print(f"  棋子数: X(己方)={num_x}, O(对手)={num_o}, 差值={num_x-num_o}")
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
            # 提取组合名称
            parts = desc.split(',')
            if len(parts) > 0:
                combo_part = parts[0].replace('Win ', '').strip()
                win_errors[combo_part] += 1

        if len(win_errors) > 0:
            for combo, count in sorted(win_errors.items(), key=lambda x: x[1], reverse=True):
                print(f"  {combo}: {count} 次错误")
        else:
            print("  无错误 ✓")

        # 分析防守案例的错误模式
        print("\n2. 防守案例错误分布:")
        lose_errors = defaultdict(int)
        for board, correct_action, predicted_action, desc in lose_results['failed_cases']:
            parts = desc.split(',')
            if len(parts) > 0:
                combo_part = parts[0].replace('Block ', '').strip()
                lose_errors[combo_part] += 1

        if len(lose_errors) > 0:
            for combo, count in sorted(lose_errors.items(), key=lambda x: x[1], reverse=True):
                print(f"  {combo}: {count} 次错误")
        else:
            print("  无错误 ✓")

    def run_full_test(self, verbose: bool = False, max_display: int = 5):
        """运行完整的枚举测试"""
        print("\n" + "=" * 70)
        print("PPO 模型 - 暴力枚举 Winning/Losing 条件测试（修复版）")
        print("=" * 70)
        print(f"模型: {self.model_path}\n")

        # 生成所有测试案例
        winning_cases, defending_cases = self.generate_all_test_cases()

        # 测试1: 获胜案例
        win_results = self.test_cases(winning_cases, "测试 1: 一步获胜（Winning Cases）",
                                     verbose=verbose, max_display=max_display)

        # 测试2: 防守案例
        lose_results = self.test_cases(defending_cases, "\n测试 2: 一步防守（Defending Cases）",
                                      verbose=verbose, max_display=max_display)

        # 错误模式分析
        self.analyze_error_patterns(win_results, lose_results)

        # 综合评估
        print("\n" + "=" * 70)
        print("综合评估")
        print("=" * 70)

        total_cases = win_results['total'] + lose_results['total']
        total_correct = win_results['correct'] + lose_results['correct']
        total_accuracy = total_correct / total_cases * 100 if total_cases > 0 else 0

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
    parser = argparse.ArgumentParser(description='暴力枚举测试 PPO 模型的 Winning/Losing 决策能力（修复版）')
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--verbose', action='store_true', help='显示详细测试过程（包括所有失败案例）')
    parser.add_argument('--max-display', type=int, default=10, help='最多显示多少个失败案例（当不使用verbose时）')

    args = parser.parse_args()

    # 创建测试器
    tester = ExhaustiveWinLoseTest(args.model)

    # 运行完整测试
    results = tester.run_full_test(verbose=args.verbose, max_display=args.max_display)

    print("\n测试完成！")


if __name__ == "__main__":
    main()
