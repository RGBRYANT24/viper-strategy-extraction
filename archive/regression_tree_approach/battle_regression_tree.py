"""
测试回归树模型（Q值树）与分类树的对比
展示回归树如何通过Q值自然避免非法动作
"""

import argparse
import numpy as np
import joblib
from battle_nn_vs_tree import (
    TicTacToeBattleEnv,
    NeuralNetPlayer,
    MinMaxPlayer,
    battle_two_players,
    print_battle_results
)


class RegressionTreePlayer:
    """
    回归树玩家（输出Q值）

    核心特点：
    - 输出9个Q值而不是单个动作
    - 自动选择合法动作中Q值最高的
    - 100%避免非法动作
    """

    def __init__(self, model_path, debug=False, explain=False):
        """
        Args:
            model_path: 回归树模型路径（.joblib）
            debug: 是否输出调试信息
            explain: 是否输出详细解释
        """
        self.model = joblib.load(model_path)
        self.debug = debug
        self.explain = explain
        self.predict_count = 0

        print(f"已加载回归树模型: {model_path}")
        print(f"  模型类型: {type(self.model)}")

        # 检查是否是MultiOutputRegressor
        if hasattr(self.model, 'estimators_'):
            print(f"  输出维度: {len(self.model.estimators_)}")
            print(f"  基础树类型: {type(self.model.estimators_[0])}")

    def predict(self, obs, player_id=1):
        """
        预测动作（使用Q值 + 合法动作掩码）

        Args:
            obs: 棋盘状态（从当前环境视角）
            player_id: 当前玩家ID（1=X, -1=O）

        Returns:
            action: 选择的动作
        """
        # 视角转换（与分类树相同）
        if player_id == -1:
            obs_transformed = -obs
        else:
            obs_transformed = obs

        obs_reshaped = obs_transformed.reshape(1, -1)

        # 预测Q值（9个值）
        q_values = self.model.predict(obs_reshaped)[0]

        # 获取合法动作
        legal_actions = np.where(obs == 0)[0]

        # 如果没有合法动作（不应该发生）
        if len(legal_actions) == 0:
            print("警告：没有合法动作！")
            return 0

        # 只在合法动作中选择Q值最高的
        legal_q_values = q_values[legal_actions]
        best_legal_idx = np.argmax(legal_q_values)
        action = legal_actions[best_legal_idx]

        # 调试/解释输出
        self.predict_count += 1
        if (self.debug or self.explain) and self.predict_count <= 5:
            self._print_decision_explanation(
                obs, obs_transformed, q_values, legal_actions, action, player_id
            )

        return action

    def _print_decision_explanation(self, obs, obs_transformed, q_values,
                                    legal_actions, final_action, player_id):
        """打印决策解释"""
        print(f"\n{'='*60}")
        print(f"回归树决策解释 #{self.predict_count}")
        print(f"{'='*60}")

        print(f"玩家ID: {player_id} ({'X' if player_id == 1 else 'O'})")
        print(f"原始棋盘: {obs}")

        if player_id == -1:
            print(f"转换棋盘: {obs_transformed}")

        print(f"\n所有位置的Q值预测:")
        print("  位置: ", end="")
        for i in range(9):
            print(f"{i:6}", end=" ")
        print()
        print("  Q值: ", end=" ")
        for i in range(9):
            print(f"{q_values[i]:6.3f}", end=" ")
        print()
        print("  状态: ", end="")
        for i in range(9):
            if i in legal_actions:
                print(f"{'合法':>6}", end=" ")
            else:
                print(f"{'非法':>6}", end=" ")
        print()

        print(f"\n合法动作: {legal_actions}")
        print(f"合法动作的Q值:")
        for action in legal_actions:
            marker = " ← 最终选择" if action == final_action else ""
            print(f"  动作 {action}: Q = {q_values[action]:.3f}{marker}")

        print(f"\n最终选择: 动作 {final_action} (Q = {q_values[final_action]:.3f})")

        # 显示被过滤的高Q值非法动作
        illegal_actions = [i for i in range(9) if i not in legal_actions]
        if illegal_actions:
            print(f"\n被过滤的非法动作:")
            for action in illegal_actions:
                print(f"  动作 {action}: Q = {q_values[action]:.3f} (已被占用)")

        print(f"{'='*60}\n")

    def get_q_values(self, obs, player_id=1):
        """
        获取Q值（用于分析）

        Returns:
            q_values: 9个位置的Q值
        """
        if player_id == -1:
            obs_transformed = -obs
        else:
            obs_transformed = obs

        obs_reshaped = obs_transformed.reshape(1, -1)
        return self.model.predict(obs_reshaped)[0]


def compare_classification_vs_regression(classification_path, regression_path, n_games=10):
    """
    对比分类树和回归树的表现

    Args:
        classification_path: 分类树模型路径
        regression_path: 回归树模型路径
        n_games: 对战局数
    """
    from battle_nn_vs_tree import DecisionTreePlayer

    print("\n" + "="*80)
    print("分类树 vs 回归树对比测试")
    print("="*80)

    # 加载两个模型
    print("\n加载模型...")
    try:
        clf_tree = DecisionTreePlayer(classification_path, debug=False)
        print("✓ 分类树加载成功")
    except Exception as e:
        print(f"✗ 分类树加载失败: {e}")
        clf_tree = None

    try:
        reg_tree = RegressionTreePlayer(regression_path, debug=False, explain=False)
        print("✓ 回归树加载成功")
    except Exception as e:
        print(f"✗ 回归树加载失败: {e}")
        return

    # 测试1: 回归树 vs MinMax
    print("\n" + "="*80)
    print("测试1: 回归树 vs MinMax（理论上应该0非法动作）")
    print("="*80)
    minmax = MinMaxPlayer()
    results1 = battle_two_players(reg_tree, minmax, n_games=n_games, verbose=False)
    print_battle_results(results1, "回归树", "MinMax")

    if results1['player1_illegal'] == 0:
        print("✓ 成功：回归树没有非法动作！")
    else:
        print(f"✗ 失败：回归树仍有 {results1['player1_illegal']} 次非法动作")

    # 测试2: 如果分类树存在，对比两者
    if clf_tree is not None:
        print("\n" + "="*80)
        print("测试2: 分类树 vs 回归树")
        print("="*80)
        results2 = battle_two_players(clf_tree, reg_tree, n_games=n_games, verbose=False)
        print_battle_results(results2, "分类树", "回归树")

        print("\n对比总结:")
        print(f"  分类树非法动作: {results2['player1_illegal']} / {n_games}")
        print(f"  回归树非法动作: {results2['player2_illegal']} / {n_games}")

        if results2['player2_illegal'] < results2['player1_illegal']:
            print("  ✓ 回归树的非法动作更少！")
        elif results2['player2_illegal'] == 0 and results2['player1_illegal'] > 0:
            print("  ✓ 回归树完全避免了非法动作！")


def test_regression_tree_single_game(model_path):
    """
    单局详细测试，展示回归树的Q值预测过程

    Args:
        model_path: 回归树模型路径
    """
    print("\n" + "="*80)
    print("回归树单局详细测试")
    print("="*80)

    env = TicTacToeBattleEnv()
    reg_tree = RegressionTreePlayer(model_path, debug=True, explain=True)
    minmax = MinMaxPlayer()

    print("\n测试: 回归树(X) vs MinMax(O)")
    print("-"*80)

    obs = env.reset()
    step = 0

    while not env.done and step < 9:
        step += 1

        if env.current_player == 1:
            current_agent = reg_tree
            agent_name = "回归树 (X)"
            player_id = 1
        else:
            current_agent = minmax
            agent_name = "MinMax (O)"
            player_id = -1

        print(f"\n步骤 {step}: {agent_name} 的回合")
        print("当前棋盘:")
        env.render()

        # 预测动作
        if isinstance(current_agent, RegressionTreePlayer):
            action = current_agent.predict(obs, player_id=player_id)
        else:
            action = current_agent.predict(obs)

        print(f"\n{agent_name} 选择动作: {action}")

        # 执行动作
        obs, reward, done, info = env.step(action)

        if done:
            print(f"\n游戏结束!")
            env.render()
            if 'illegal_move' in info:
                print(f"原因: 非法移动")
            elif 'winner' in info:
                print(f"原因: {'X' if info['winner'] == 1 else 'O'} 获胜")
            elif 'draw' in info:
                print(f"原因: 平局")
            break

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="回归树模型测试")

    parser.add_argument("--regression-path", type=str,
                       default="log/viper_TicTacToe-v0_all-leaves_10_regression.joblib",
                       help="回归树模型路径")
    parser.add_argument("--classification-path", type=str,
                       default="log/viper_TicTacToe-v0_all-leaves_10.joblib",
                       help="分类树模型路径（用于对比）")

    parser.add_argument("--mode", type=str, default="compare",
                       choices=['single', 'compare', 'test'],
                       help="测试模式: single=单局详细测试, compare=对比测试, test=快速测试")

    parser.add_argument("--n-games", type=int, default=20,
                       help="对战局数")

    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")

    args = parser.parse_args()

    # 设置随机种子
    np.random.seed(args.seed)

    if args.mode == 'single':
        # 单局详细测试
        test_regression_tree_single_game(args.regression_path)

    elif args.mode == 'compare':
        # 对比测试
        compare_classification_vs_regression(
            args.classification_path,
            args.regression_path,
            n_games=args.n_games
        )

    elif args.mode == 'test':
        # 快速测试
        print("\n快速测试：回归树是否能避免非法动作\n")
        reg_tree = RegressionTreePlayer(args.regression_path)
        minmax = MinMaxPlayer()

        results = battle_two_players(reg_tree, minmax, n_games=args.n_games, verbose=False)
        print_battle_results(results, "回归树", "MinMax")

        if results['player1_illegal'] == 0:
            print("\n✓ 成功：回归树在所有测试中都避免了非法动作！")
        else:
            print(f"\n✗ 回归树仍有 {results['player1_illegal']} 次非法动作")


if __name__ == "__main__":
    main()
