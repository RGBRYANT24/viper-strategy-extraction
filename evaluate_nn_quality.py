"""
神经网络质量全面评估工具
判断TicTacToe策略是否达到最优
"""

import argparse
import numpy as np
from stable_baselines3 import DQN
import gymnasium as gym
from itertools import permutations
import torch

# 导入环境注册
import gym_env


class TicTacToeNNEvaluator:
    """TicTacToe神经网络质量评估器"""

    def __init__(self, model_path):
        self.model = DQN.load(model_path)
        self.win_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # 行
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # 列
            [0, 4, 8], [2, 4, 6]              # 对角线
        ]

    def evaluate_critical_positions(self):
        """评估关键局面识别能力（最优策略必须正确的局面）"""
        print("=" * 70)
        print("评估 1: 关键局面识别能力")
        print("=" * 70)

        test_cases = [
            # (棋盘状态, 正确动作列表, 描述)
            # 1. 立即获胜机会
            ([1, 1, 0, 0, 0, 0, 0, 0, 0], [2], "立即获胜-横"),
            ([1, 0, 0, 1, 0, 0, 0, 0, 0], [7], "立即获胜-竖"),
            ([1, 0, 0, 0, 1, 0, 0, 0, 0], [8], "立即获胜-对角线"),
            ([0, 0, 1, 0, 1, 0, 1, 0, 0], [3], "立即获胜-反对角线"),

            # 2. 必须防守
            ([-1, -1, 0, 0, 0, 0, 0, 0, 0], [2], "防守获胜-横"),
            ([-1, 0, 0, -1, 0, 0, 0, 0, 0], [7], "防守获胜-竖"),
            ([-1, 0, 0, 0, -1, 0, 0, 0, 0], [8], "防守获胜-对角线"),

            # 3. 创造双威胁（叉攻）
            ([1, 0, 0, 0, 0, 0, 0, 0, 0], [4, 2, 6, 8], "开局-中心或角落"),
            ([1, 0, 0, 0, 0, 0, 0, 0, 1], [4], "双角-占中心"),
            ([1, 0, 0, 0, 1, 0, 0, 0, 0], [2, 6], "中心+角落-创造叉攻"),

            # 4. 破坏对手叉攻
            ([-1, 0, 0, 0, 0, 0, 0, 0, -1], [4], "对手双角-必须占中心"),
            ([-1, 0, 0, 0, -1, 0, 0, 0, 0], [1, 3, 5, 7], "对手中心+角落-占边"),

            # 5. 空棋盘
            ([0, 0, 0, 0, 0, 0, 0, 0, 0], [4], "空棋盘-中心最优"),
        ]

        total = 0
        correct = 0

        for board, correct_actions, description in test_cases:
            obs = np.array(board, dtype=np.float32)
            action, _ = self.model.predict(obs, deterministic=True)

            is_correct = action in correct_actions
            total += 1
            if is_correct:
                correct += 1

            symbol = "✓" if is_correct else "✗"
            print(f"\n{symbol} {description}")
            print(f"  棋盘:\n{obs.reshape(3, 3)}")
            print(f"  预测: {action}, 正确: {correct_actions}")

        accuracy = correct / total * 100
        print(f"\n总体准确率: {correct}/{total} = {accuracy:.1f}%")

        if accuracy == 100:
            print("🏆 完美！所有关键局面都识别正确")
        elif accuracy >= 90:
            print("✓ 优秀！大部分关键局面识别正确")
        elif accuracy >= 70:
            print("△ 良好，但还有改进空间")
        else:
            print("⚠ 需要改进，关键局面识别率较低")

        return accuracy

    def evaluate_symmetry_consistency(self):
        """评估对称一致性（最优策略必须对称等价）"""
        print("\n" + "=" * 70)
        print("评估 2: 对称性一致性")
        print("=" * 70)

        # 测试空棋盘的8种对称变换
        print("\n[测试] 空棋盘的对称性")
        empty_board = np.zeros(9, dtype=np.float32)
        action_empty, _ = self.model.predict(empty_board, deterministic=True)

        # 中心(4)是唯一的，角落(0,2,6,8)等价，边(1,3,5,7)等价
        if action_empty == 4:
            print(f"  ✓ 选择中心({action_empty}) - 最优")
        elif action_empty in [0, 2, 6, 8]:
            print(f"  △ 选择角落({action_empty}) - 次优但可接受")
        else:
            print(f"  ✗ 选择边({action_empty}) - 不佳")

        # 测试对称局面
        print("\n[测试] 对称局面一致性")
        test_board = np.array([1, 0, -1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        print(f"  原始棋盘:\n{test_board.reshape(3, 3)}")

        # 旋转90度的等价位置映射
        rotate_90 = [6, 3, 0, 7, 4, 1, 8, 5, 2]
        rotate_180 = [8, 7, 6, 5, 4, 3, 2, 1, 0]
        rotate_270 = [2, 5, 8, 1, 4, 7, 0, 3, 6]

        transformations = [
            (test_board, "原始"),
            (test_board[rotate_90], "旋转90°"),
            (test_board[rotate_180], "旋转180°"),
            (test_board[rotate_270], "旋转270°"),
        ]

        actions = []
        for board, name in transformations:
            action, _ = self.model.predict(board, deterministic=True)
            actions.append(action)
            print(f"  {name}: 选择位置 {action}")

        # 检查是否对称等价（简化检查：至少选择类型相同）
        # 中心=4, 角落={0,2,6,8}, 边={1,3,5,7}
        def get_type(pos):
            if pos == 4:
                return "center"
            elif pos in [0, 2, 6, 8]:
                return "corner"
            else:
                return "edge"

        types = [get_type(a) for a in actions]
        if len(set(types)) == 1:
            print(f"  ✓ 对称一致性良好（都选择{types[0]}）")
            return True
        else:
            print(f"  ⚠ 对称不一致: {types}")
            return False

    def evaluate_against_perfect_play(self):
        """评估对战完美对手（MinMax）的表现"""
        print("\n" + "=" * 70)
        print("评估 3: 对战完美对手 (MinMax)")
        print("=" * 70)

        # 先手测试
        print("\n[先手] 50局 vs MinMax")
        env_x = gym.make('TicTacToe-v0', opponent_type='minmax')
        wins_x, losses_x, draws_x = 0, 0, 0

        for _ in range(50):
            obs, _ = env_x.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env_x.step(action)
                done = terminated or truncated
                if done:
                    if reward > 0:
                        wins_x += 1
                    elif reward < 0:
                        losses_x += 1
                    else:
                        draws_x += 1

        env_x.close()

        print(f"  胜: {wins_x} ({wins_x/50*100:.1f}%)")
        print(f"  平: {draws_x} ({draws_x/50*100:.1f}%)")
        print(f"  负: {losses_x} ({losses_x/50*100:.1f}%)")

        # 后手测试（需要修改环境让MinMax先手）
        print("\n[后手] 通过翻转模拟后手对战")
        # 使用delta-selfplay环境模拟后手
        from gym_env.tictactoe_delta_selfplay import TicTacToeDeltaSelfPlayEnv
        from gym_env.policies import MinMaxPlayerPolicy
        from gymnasium import spaces

        obs_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.float32)
        act_space = spaces.Discrete(9)
        minmax_pool = [MinMaxPlayerPolicy(obs_space, act_space)]

        env_o = TicTacToeDeltaSelfPlayEnv(
            baseline_pool=minmax_pool,
            learned_pool=None,
            play_as_o_prob=1.0  # 强制后手
        )

        wins_o, losses_o, draws_o = 0, 0, 0
        for _ in range(50):
            obs, _ = env_o.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env_o.step(action)
                done = terminated or truncated
                if done:
                    if reward > 0:
                        wins_o += 1
                    elif reward < 0:
                        losses_o += 1
                    else:
                        draws_o += 1

        env_o.close()

        print(f"  胜: {wins_o} ({wins_o/50*100:.1f}%)")
        print(f"  平: {draws_o} ({draws_o/50*100:.1f}%)")
        print(f"  负: {losses_o} ({losses_o/50*100:.1f}%)")

        # 总体评估
        total_draws = draws_x + draws_o
        total_losses = losses_x + losses_o
        total_games = 100

        print(f"\n总体 (100局):")
        print(f"  平局率: {total_draws/total_games*100:.1f}%")
        print(f"  输掉率: {total_losses/total_games*100:.1f}%")

        # 判断是否最优
        is_optimal = total_draws >= 95 and total_losses == 0

        if is_optimal:
            print(f"  🏆 达到最优！无输局且平局率≥95%")
        elif total_draws >= 90:
            print(f"  ✓ 接近最优！平局率≥90%")
        elif total_draws >= 80:
            print(f"  △ 良好，但还有提升空间")
        else:
            print(f"  ⚠ 需要改进")

        return is_optimal, total_draws / total_games

    def evaluate_q_value_quality(self):
        """评估Q值质量"""
        print("\n" + "=" * 70)
        print("评估 4: Q值质量分析")
        print("=" * 70)

        test_cases = [
            # (棋盘, 描述, 应该高Q值的动作, 应该低Q值的动作)
            (
                np.array([1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
                "立即获胜机会",
                [2],  # 获胜动作
                [3, 4, 5, 6, 7, 8]  # 其他动作
            ),
            (
                np.array([-1, -1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
                "必须防守",
                [2],  # 防守动作
                [3, 4, 5, 6, 7, 8]  # 不防守会输
            ),
            (
                np.zeros(9, dtype=np.float32),
                "空棋盘",
                [4],  # 中心
                [1, 3, 5, 7]  # 边（最差）
            ),
        ]

        for obs, description, high_q_actions, low_q_actions in test_cases:
            print(f"\n[{description}]")
            print(f"  棋盘:\n{obs.reshape(3, 3)}")

            # 获取Q值
            obs_tensor = self.model.policy.obs_to_tensor(obs)[0]
            with torch.no_grad():
                q_values = self.model.policy.q_net(obs_tensor).cpu().numpy()[0]

            legal_actions = np.where(obs == 0)[0]
            q_legal = {i: q_values[i] for i in legal_actions}

            print(f"  合法动作Q值: {{{', '.join(f'{k}:{v:.3f}' for k, v in q_legal.items())}}}")

            # 检查最高Q值是否在正确动作中
            best_action = max(q_legal, key=q_legal.get)
            best_q = q_legal[best_action]

            if best_action in high_q_actions:
                print(f"  ✓ 最高Q值在正确动作 {best_action} (Q={best_q:.3f})")
            else:
                print(f"  ✗ 最高Q值在错误动作 {best_action} (Q={best_q:.3f})")
                print(f"     正确动作应该是: {high_q_actions}")

            # 检查Q值差异
            if high_q_actions[0] in q_legal and low_q_actions:
                low_q_in_legal = [a for a in low_q_actions if a in q_legal]
                if low_q_in_legal:
                    high_q = q_legal[high_q_actions[0]]
                    low_q = np.mean([q_legal[a] for a in low_q_in_legal])
                    diff = high_q - low_q

                    if diff > 0.5:
                        print(f"  ✓ Q值区分明显 (Δ={diff:.3f})")
                    elif diff > 0.1:
                        print(f"  △ Q值有区分 (Δ={diff:.3f})")
                    else:
                        print(f"  ⚠ Q值区分不足 (Δ={diff:.3f})")

    def evaluate_exploration_vs_exploitation(self):
        """评估探索vs利用的平衡"""
        print("\n" + "=" * 70)
        print("评估 5: 策略稳定性（确定性vs随机性）")
        print("=" * 70)

        obs = np.zeros(9, dtype=np.float32)

        # 多次预测，检查一致性
        print("\n[测试] 空棋盘预测10次（确定性）")
        actions_det = [self.model.predict(obs, deterministic=True)[0] for _ in range(10)]
        unique_det = set(actions_det)

        print(f"  预测结果: {actions_det}")
        print(f"  唯一值: {unique_det}")

        if len(unique_det) == 1:
            print(f"  ✓ 完全确定性 - 策略稳定")
        else:
            print(f"  ⚠ 存在随机性 - 可能探索率未归零")

        # 检查当前探索率
        print(f"\n[参数] 当前探索率: {self.model.exploration_rate:.6f}")
        if self.model.exploration_rate <= 0.05:
            print(f"  ✓ 探索率已收敛到最低值")
        else:
            print(f"  ⚠ 探索率仍较高，可能影响确定性")

    def evaluate_comprehensive(self):
        """综合评估"""
        print("\n" + "🎯" * 35)
        print("TicTacToe 神经网络质量综合评估")
        print("🎯" * 35 + "\n")

        print(f"模型路径: {self.model.num_timesteps} 步训练")
        print(f"网络结构: {self.model.policy.net_arch}")
        print()

        # 执行所有评估
        critical_accuracy = self.evaluate_critical_positions()
        symmetry_ok = self.evaluate_symmetry_consistency()
        is_optimal, draw_rate = self.evaluate_against_perfect_play()
        self.evaluate_q_value_quality()
        self.evaluate_exploration_vs_exploitation()

        # 最终判断
        print("\n" + "=" * 70)
        print("最终判断: 策略是否最优？")
        print("=" * 70)

        criteria = {
            "关键局面识别": critical_accuracy >= 90,
            "对称性一致": symmetry_ok,
            "vs MinMax平局率": draw_rate >= 0.95,
            "无输局": is_optimal,
        }

        print("\n评估标准:")
        for criterion, passed in criteria.items():
            symbol = "✓" if passed else "✗"
            print(f"  {symbol} {criterion}")

        all_passed = all(criteria.values())

        print("\n" + "=" * 70)
        if all_passed:
            print("🏆 恭喜！你的神经网络已达到TicTacToe的最优策略！")
            print()
            print("证据:")
            print(f"  1. 关键局面识别准确率: {critical_accuracy:.1f}%")
            print(f"  2. vs MinMax平局率: {draw_rate*100:.1f}%")
            print(f"  3. 对称性一致: {'是' if symmetry_ok else '否'}")
            print()
            print("这意味着:")
            print("  • 先手: 永不会输，除非自己犯错")
            print("  • 后手: 永不会输给完美对手")
            print("  • 所有关键战术点（获胜/防守/叉攻）都能识别")
            print()
            print("下一步建议:")
            print("  1. 使用VIPER提取决策树")
            print("  2. 可视化策略规则")
            print("  3. 形式化验证")

        elif draw_rate >= 0.9:
            print("✓ 你的神经网络接近最优策略！")
            print()
            print(f"  • vs MinMax平局率: {draw_rate*100:.1f}% (目标: ≥95%)")
            print(f"  • 关键局面准确率: {critical_accuracy:.1f}% (目标: ≥90%)")
            print()
            print("改进建议:")
            if critical_accuracy < 90:
                print("  • 继续训练，提高关键局面识别")
            if draw_rate < 0.95:
                print("  • 增加MinMax基准对手的训练比例")

        else:
            print("△ 策略良好，但距离最优还有距离")
            print()
            print("需要改进:")
            for criterion, passed in criteria.items():
                if not passed:
                    print(f"  • {criterion}")

        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='评估TicTacToe神经网络质量')
    parser.add_argument('--model', type=str,
                        default='log/oracle_TicTacToe_delta_selfplay.zip',
                        help='模型路径')
    args = parser.parse_args()

    evaluator = TicTacToeNNEvaluator(args.model)
    evaluator.evaluate_comprehensive()


if __name__ == "__main__":
    main()
