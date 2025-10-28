"""
综合评估脚本：测试 PPO 模型是否学到了好的 Tic-Tac-Toe 策略
评估维度：
1. 对抗测试（vs Random, MinMax）
2. 战术知识测试（获胜、防守、双威胁）
3. 开局质量分析
4. 策略一致性测试（对称性、确定性）
5. 动作价值分析
"""

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
import torch
import sys
import os
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gym_env


def mask_fn(env):
    """返回 action mask"""
    if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'board'):
        board = env.unwrapped.board
    elif hasattr(env, 'board'):
        board = env.board
    elif hasattr(env, 'env') and hasattr(env.env, 'board'):
        board = env.env.board
    else:
        return np.ones(9, dtype=np.int8)

    mask = (board == 0).astype(np.int8)
    return mask


class TicTacToeEvaluator:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.env = None

    def load_model(self):
        """加载模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        # 创建环境
        self.env = gym.make('TicTacToe-v0', opponent_type='random')
        self.env = Monitor(self.env)
        self.env = ActionMasker(self.env, mask_fn)

        # 加载模型
        self.model = MaskablePPO.load(self.model_path, env=self.env)
        print(f"✓ 模型加载成功: {self.model_path}\n")

    def test_against_opponents(self, num_games: int = 100) -> Dict[str, Dict]:
        """测试1: 对抗不同对手"""
        print("=" * 70)
        print("测试 1: 对抗测试")
        print("=" * 70)

        results = {}
        opponents = ['random', 'minmax']

        for opponent in opponents:
            print(f"\n对战 {opponent.upper()} 对手 ({num_games} 局)...")

            # 创建对应环境
            env = gym.make('TicTacToe-v0', opponent_type=opponent)
            env = Monitor(env)
            env = ActionMasker(env, mask_fn)

            wins, losses, draws, illegal_moves = 0, 0, 0, 0
            total_steps = []

            for i in range(num_games):
                obs, _ = env.reset()
                done = False
                step_count = 0

                while not done:
                    mask = mask_fn(env)
                    action, _ = self.model.predict(obs, deterministic=True, action_masks=mask)

                    # 检查非法动作
                    if mask[action] == 0:
                        illegal_moves += 1

                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    step_count += 1

                    if done:
                        total_steps.append(step_count)
                        if 'illegal_move' in info and info['illegal_move']:
                            losses += 1
                        elif reward > 0:
                            wins += 1
                        elif reward < 0:
                            losses += 1
                        else:
                            draws += 1

            win_rate = wins / num_games * 100
            loss_rate = losses / num_games * 100
            draw_rate = draws / num_games * 100
            avg_steps = np.mean(total_steps)

            print(f"  胜率: {win_rate:.1f}% ({wins}/{num_games})")
            print(f"  负率: {loss_rate:.1f}% ({losses}/{num_games})")
            print(f"  平局率: {draw_rate:.1f}% ({draws}/{num_games})")
            print(f"  非法移动: {illegal_moves}")
            print(f"  平均步数: {avg_steps:.1f}")

            # 评估标准
            if opponent == 'random':
                if win_rate >= 95:
                    grade = "优秀 ✓✓"
                elif win_rate >= 85:
                    grade = "良好 ✓"
                elif win_rate >= 70:
                    grade = "及格"
                else:
                    grade = "不及格 ✗"
                print(f"  评级: {grade} (期望 >95%)")
            else:  # minmax
                if win_rate >= 50:
                    grade = "优秀 ✓✓ (能战胜最优对手!)"
                elif draw_rate >= 80:
                    grade = "良好 ✓"
                elif draw_rate >= 50:
                    grade = "及格"
                else:
                    grade = "不及格 ✗"
                print(f"  评级: {grade} (期望平局率 >80%)")

            results[opponent] = {
                'wins': wins,
                'losses': losses,
                'draws': draws,
                'illegal_moves': illegal_moves,
                'win_rate': win_rate,
                'avg_steps': avg_steps
            }

            env.close()

        return results

    def test_tactical_knowledge(self) -> Dict[str, bool]:
        """测试2: 战术知识测试"""
        print("\n" + "=" * 70)
        print("测试 2: 战术知识测试")
        print("=" * 70)

        test_cases = [
            # (棋盘状态, 最优动作, 描述)
            # X = 1, O = -1, Empty = 0
            # 立即获胜测试
            ([1, 1, 0, -1, -1, 0, 0, 0, 0], 2, "立即获胜: 完成第一行"),
            ([1, -1, 0, 1, -1, 0, 0, 0, 0], 6, "立即获胜: 完成第一列"),
            ([1, -1, -1, 0, 1, 0, 0, 0, 0], 8, "立即获胜: 完成主对角线"),
            ([0, 0, 1, -1, 1, -1, 0, 0, 0], 6, "立即获胜: 完成副对角线"),

            # 阻止对手获胜测试
            ([-1, -1, 0, 1, 0, 0, 0, 0, 0], 2, "阻止对手: 防守第一行"),
            ([-1, 1, 0, -1, 0, 0, 0, 1, 0], 6, "阻止对手: 防守第一列"),
            ([-1, 0, 1, 0, -1, 0, 1, 0, 0], 8, "阻止对手: 防守主对角线"),

            # 制造双威胁
            ([1, 0, 0, 0, 0, 0, 0, 0, 1], [2, 4, 6], "双威胁: 角-角对角，走边或中心"),
            ([1, 0, 0, 0, 1, 0, 0, 0, 0], [1, 3, 5, 7], "中心控制: 扩展威胁"),
        ]

        results = {}
        correct_count = 0

        for i, (board, optimal_actions, description) in enumerate(test_cases):
            # 支持单个动作或多个可接受动作
            if isinstance(optimal_actions, int):
                optimal_actions = [optimal_actions]

            board_array = np.array(board, dtype=np.float32)
            mask = (board_array == 0).astype(np.int8)

            # 获取模型预测
            action, _ = self.model.predict(board_array, deterministic=True, action_masks=mask)
            action = int(action)

            # 检查是否正确
            is_correct = action in optimal_actions
            correct_count += is_correct

            # 可视化棋盘
            board_str = self._visualize_board(board)

            status = "✓" if is_correct else "✗"
            print(f"\n测试 {i+1}: {description} {status}")
            print(board_str)
            print(f"  模型选择: {action}")
            print(f"  最优动作: {optimal_actions}")

            if not is_correct:
                # 显示动作概率分布
                probs = self._get_action_probs(board_array, mask)
                print(f"  动作概率: {probs}")

            results[description] = is_correct

        accuracy = correct_count / len(test_cases) * 100
        print(f"\n战术知识准确率: {correct_count}/{len(test_cases)} ({accuracy:.1f}%)")

        if accuracy >= 90:
            grade = "优秀 ✓✓"
        elif accuracy >= 70:
            grade = "良好 ✓"
        elif accuracy >= 50:
            grade = "及格"
        else:
            grade = "不及格 ✗"
        print(f"评级: {grade}")

        return results

    def test_opening_quality(self, num_samples: int = 100) -> Dict:
        """测试3: 开局质量分析"""
        print("\n" + "=" * 70)
        print("测试 3: 开局质量分析")
        print("=" * 70)

        # 统计首步选择
        first_moves = defaultdict(int)

        empty_board = np.zeros(9, dtype=np.float32)
        mask = np.ones(9, dtype=np.int8)

        for _ in range(num_samples):
            action, _ = self.model.predict(empty_board, deterministic=False, action_masks=mask)
            first_moves[int(action)] += 1

        # 分类统计
        center = 4
        corners = [0, 2, 6, 8]
        edges = [1, 3, 5, 7]

        center_count = first_moves[center]
        corner_count = sum(first_moves[pos] for pos in corners)
        edge_count = sum(first_moves[pos] for pos in edges)

        print(f"\n首步选择分布 ({num_samples} 次采样):")
        print(f"  中心 (位置4): {center_count} ({center_count/num_samples*100:.1f}%)")
        print(f"  角落 (0,2,6,8): {corner_count} ({corner_count/num_samples*100:.1f}%)")
        print(f"  边 (1,3,5,7): {edge_count} ({edge_count/num_samples*100:.1f}%)")

        # 显示具体分布
        print(f"\n详细分布:")
        board_visual = [
            f"{first_moves[0]:3d}", f"{first_moves[1]:3d}", f"{first_moves[2]:3d}",
            f"{first_moves[3]:3d}", f"{first_moves[4]:3d}", f"{first_moves[5]:3d}",
            f"{first_moves[6]:3d}", f"{first_moves[7]:3d}", f"{first_moves[8]:3d}",
        ]
        print(f"  {board_visual[0]} | {board_visual[1]} | {board_visual[2]}")
        print(f"  -----------")
        print(f"  {board_visual[3]} | {board_visual[4]} | {board_visual[5]}")
        print(f"  -----------")
        print(f"  {board_visual[6]} | {board_visual[7]} | {board_visual[8]}")

        # 评估开局质量
        # 理论上：中心 ≥ 角落 > 边
        if center_count >= corner_count >= edge_count:
            if center_count > num_samples * 0.3:
                grade = "优秀 ✓✓"
            else:
                grade = "良好 ✓"
        elif corner_count > edge_count:
            grade = "及格"
        else:
            grade = "不及格 ✗"

        print(f"\n评级: {grade} (期望: 中心 > 角落 > 边)")

        return {
            'center': center_count,
            'corners': corner_count,
            'edges': edge_count,
            'distribution': dict(first_moves)
        }

    def test_symmetry(self, num_samples: int = 10) -> Dict:
        """测试4: 对称性测试"""
        print("\n" + "=" * 70)
        print("测试 4: 策略对称性测试")
        print("=" * 70)

        # 测试旋转对称性
        test_boards = [
            # 简单开局
            [1, 0, 0, 0, 0, 0, 0, 0, 0],  # X 在左上角
            # 复杂局面
            [1, -1, 0, 0, 1, 0, 0, 0, -1],
        ]

        symmetry_violations = 0
        total_tests = 0

        for board in test_boards:
            board_array = np.array(board, dtype=np.float32)

            # 获取原始棋盘的动作概率
            mask = (board_array == 0).astype(np.int8)
            orig_probs = self._get_action_probs(board_array, mask)
            orig_action, _ = self.model.predict(board_array, deterministic=True, action_masks=mask)

            print(f"\n原始棋盘:")
            print(self._visualize_board(board))
            print(f"  选择动作: {orig_action}")

            # 测试 90° 旋转
            rotated_board = self._rotate_board(board_array, k=1)
            rotated_mask = (rotated_board == 0).astype(np.int8)
            rotated_action, _ = self.model.predict(rotated_board, deterministic=True, action_masks=rotated_mask)
            expected_action = self._rotate_action(int(orig_action), k=1)

            total_tests += 1
            if rotated_action != expected_action:
                symmetry_violations += 1
                print(f"  旋转90°: ✗ (预测 {rotated_action}, 期望 {expected_action})")
            else:
                print(f"  旋转90°: ✓")

        accuracy = (total_tests - symmetry_violations) / total_tests * 100
        print(f"\n对称性准确率: {total_tests - symmetry_violations}/{total_tests} ({accuracy:.1f}%)")

        return {
            'violations': symmetry_violations,
            'total': total_tests,
            'accuracy': accuracy
        }

    def test_determinism(self, num_samples: int = 10) -> bool:
        """测试5: 确定性测试"""
        print("\n" + "=" * 70)
        print("测试 5: 确定性一致性测试")
        print("=" * 70)

        test_boards = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, -1, 0, 0, 0, 0],
            [1, -1, 1, 0, -1, 0, 0, 0, 0],
        ]

        all_consistent = True

        for board in test_boards:
            board_array = np.array(board, dtype=np.float32)
            mask = (board_array == 0).astype(np.int8)

            # 多次预测
            actions = []
            for _ in range(num_samples):
                action, _ = self.model.predict(board_array, deterministic=True, action_masks=mask)
                actions.append(int(action))

            # 检查是否一致
            is_consistent = len(set(actions)) == 1
            all_consistent &= is_consistent

            status = "✓" if is_consistent else "✗"
            print(f"\n棋盘: {board} {status}")
            if not is_consistent:
                print(f"  动作序列: {actions}")

        if all_consistent:
            print(f"\n✓ 所有测试都保持确定性")
        else:
            print(f"\n✗ 存在不确定性")

        return all_consistent

    def analyze_action_values(self) -> None:
        """测试6: 动作价值分析"""
        print("\n" + "=" * 70)
        print("测试 6: 动作价值分析")
        print("=" * 70)

        test_cases = [
            ([0, 0, 0, 0, 0, 0, 0, 0, 0], "空棋盘"),
            ([1, 0, 0, 0, 0, 0, 0, 0, 0], "X 在角落"),
            ([0, 0, 0, 0, 1, 0, 0, 0, 0], "X 在中心"),
            ([1, 1, 0, -1, -1, 0, 0, 0, 0], "关键决策"),
        ]

        for board, description in test_cases:
            board_array = np.array(board, dtype=np.float32)
            mask = (board_array == 0).astype(np.int8)

            # 获取动作概率
            probs = self._get_action_probs(board_array, mask)

            # 获取最优动作
            best_action = np.argmax(probs)
            best_prob = probs[best_action]

            # 获取次优动作
            probs_copy = probs.copy()
            probs_copy[best_action] = -1
            second_best_action = np.argmax(probs_copy)
            second_best_prob = probs_copy[second_best_action]

            print(f"\n{description}:")
            print(self._visualize_board(board))
            print(f"  最优动作: {best_action} (概率: {best_prob:.3f})")
            print(f"  次优动作: {second_best_action} (概率: {second_best_prob:.3f})")
            print(f"  概率差距: {best_prob - second_best_prob:.3f}")

            # 显示前3个动作
            top_3 = np.argsort(probs)[-3:][::-1]
            print(f"  前3动作: {[f'{a}({probs[a]:.2f})' for a in top_3]}")

    def _get_action_probs(self, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """获取动作概率分布"""
        obs_tensor = torch.tensor(obs).float().unsqueeze(0).to(self.model.device)
        mask_tensor = torch.tensor(mask).unsqueeze(0).to(self.model.device)

        with torch.no_grad():
            distribution = self.model.policy.get_distribution(obs_tensor, action_masks=mask_tensor)
            probs = distribution.distribution.probs.cpu().numpy()[0]

        return probs

    def _visualize_board(self, board: List[int]) -> str:
        """可视化棋盘"""
        symbols = {0: '.', 1: 'X', -1: 'O'}
        lines = []
        for i in range(0, 9, 3):
            row = [symbols[board[j]] for j in range(i, i+3)]
            lines.append(f"  {row[0]} | {row[1]} | {row[2]}")
            if i < 6:
                lines.append(f"  ---------")
        return "\n".join(lines)

    def _rotate_board(self, board: np.ndarray, k: int = 1) -> np.ndarray:
        """旋转棋盘 k*90 度"""
        board_2d = board.reshape(3, 3)
        rotated = np.rot90(board_2d, k=k)
        return rotated.flatten()

    def _rotate_action(self, action: int, k: int = 1) -> int:
        """旋转动作索引"""
        # 将 action 映射到 2D 坐标
        row, col = action // 3, action % 3

        # 旋转 k 次
        for _ in range(k):
            row, col = col, 2 - row

        return row * 3 + col

    def run_full_evaluation(self, num_games: int = 100):
        """运行完整评估"""
        print("\n" + "=" * 70)
        print("PPO 模型策略质量综合评估")
        print("=" * 70)
        print(f"模型: {self.model_path}\n")

        self.load_model()

        # 运行所有测试
        opponent_results = self.test_against_opponents(num_games)
        tactical_results = self.test_tactical_knowledge()
        opening_results = self.test_opening_quality()
        symmetry_results = self.test_symmetry()
        determinism_ok = self.test_determinism()
        self.analyze_action_values()

        # 综合评分
        print("\n" + "=" * 70)
        print("综合评估总结")
        print("=" * 70)

        scores = []

        # 1. 对抗测试评分
        random_win_rate = opponent_results['random']['win_rate']
        minmax_draw_rate = opponent_results['minmax']['draws'] / num_games * 100

        if random_win_rate >= 95:
            opponent_score = 100
        elif random_win_rate >= 85:
            opponent_score = 80
        elif random_win_rate >= 70:
            opponent_score = 60
        else:
            opponent_score = 40

        scores.append(("对抗测试", opponent_score))
        print(f"\n1. 对抗测试: {opponent_score}/100")
        print(f"   vs Random: {random_win_rate:.1f}% 胜率")
        print(f"   vs MinMax: {minmax_draw_rate:.1f}% 平局率")

        # 2. 战术知识评分
        tactical_correct = sum(tactical_results.values())
        tactical_total = len(tactical_results)
        tactical_score = tactical_correct / tactical_total * 100

        scores.append(("战术知识", tactical_score))
        print(f"\n2. 战术知识: {tactical_score:.0f}/100")
        print(f"   正确率: {tactical_correct}/{tactical_total}")

        # 3. 开局质量评分
        center_pct = opening_results['center'] / 100
        corners_pct = opening_results['corners'] / 100
        edges_pct = opening_results['edges'] / 100

        if center_pct >= corners_pct >= edges_pct:
            if center_pct >= 30:
                opening_score = 100
            else:
                opening_score = 80
        elif corners_pct > edges_pct:
            opening_score = 60
        else:
            opening_score = 40

        scores.append(("开局质量", opening_score))
        print(f"\n3. 开局质量: {opening_score}/100")

        # 4. 对称性评分
        symmetry_score = symmetry_results['accuracy']
        scores.append(("策略对称性", symmetry_score))
        print(f"\n4. 策略对称性: {symmetry_score:.0f}/100")

        # 5. 确定性评分
        determinism_score = 100 if determinism_ok else 0
        scores.append(("确定性", determinism_score))
        print(f"\n5. 确定性: {determinism_score}/100")

        # 总分
        total_score = np.mean([s for _, s in scores])

        print(f"\n" + "=" * 70)
        print(f"总体评分: {total_score:.1f}/100")

        if total_score >= 90:
            print(f"总体评级: 优秀 ✓✓ - 模型学到了高质量的策略!")
        elif total_score >= 75:
            print(f"总体评级: 良好 ✓ - 模型策略基本合理")
        elif total_score >= 60:
            print(f"总体评级: 及格 - 模型有一定策略能力")
        else:
            print(f"总体评级: 不及格 ✗ - 模型策略质量不足")

        print("=" * 70)

        self.env.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='评估 PPO 模型策略质量')
    parser.add_argument('--model', type=str,
                       default='log/oracle_TicTacToe_ppo_masked.zip',
                       help='模型路径')
    parser.add_argument('--num-games', type=int, default=100,
                       help='对抗测试的游戏局数')

    args = parser.parse_args()

    evaluator = TicTacToeEvaluator(args.model)
    evaluator.run_full_evaluation(num_games=args.num_games)


if __name__ == "__main__":
    main()
