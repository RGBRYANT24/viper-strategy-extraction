"""
测试单棵分类树 + 概率掩码方案
展示如何在保持可解释性的同时避免非法动作
"""

import argparse
import numpy as np
import joblib
from battle_nn_vs_tree import (
    TicTacToeBattleEnv,
    MinMaxPlayer,
    battle_two_players,
    print_battle_results
)


class SingleTreePlayer:
    """
    单棵分类树 + 概率掩码玩家

    核心特点：
    - 单棵决策树（完整可解释性）
    - 使用predict_proba()获取概率
    - 应用合法动作掩码
    - 100%避免非法动作
    """

    def __init__(self, model_path, debug=False, explain=False):
        self.model = joblib.load(model_path)
        self.debug = debug
        self.explain = explain
        self.predict_count = 0

        print(f"已加载单棵分类树模型: {model_path}")
        print(f"  模型类型: {type(self.model)}")
        if hasattr(self.model, 'tree_'):
            print(f"  叶节点数: {self.model.tree_.n_leaves}")
            print(f"  树深度: {self.model.tree_.max_depth}")
            print(f"  类别数: {self.model.n_classes_}")

    def predict(self, obs, player_id=1):
        """
        预测动作（使用概率掩码）

        Args:
            obs: 棋盘状态
            player_id: 玩家ID（1=X, -1=O）

        Returns:
            action: 选择的动作
        """
        # 视角转换
        if player_id == -1:
            obs_transformed = -obs
        else:
            obs_transformed = obs

        obs_reshaped = obs_transformed.reshape(1, -1)

        # 获取概率分布
        action_probs = self.model.predict_proba(obs_reshaped)[0]

        # 获取合法动作
        legal_actions = np.where(obs == 0)[0]

        if len(legal_actions) == 0:
            print("警告：没有合法动作！")
            return 0

        # 应用掩码：非法动作的概率设为-inf
        masked_probs = np.full(9, -np.inf)
        masked_probs[legal_actions] = action_probs[legal_actions]

        # 选择合法动作中概率最高的
        action = np.argmax(masked_probs)

        # 调试/解释输出
        self.predict_count += 1
        if (self.debug or self.explain) and self.predict_count <= 5:
            self._print_decision_explanation(
                obs, obs_transformed, action_probs, legal_actions, action, player_id
            )

        return action

    def _print_decision_explanation(self, obs, obs_transformed, probs,
                                    legal_actions, final_action, player_id):
        """打印决策解释"""
        print(f"\n{'='*60}")
        print(f"单棵树决策解释 #{self.predict_count}")
        print(f"{'='*60}")

        print(f"玩家ID: {player_id} ({'X' if player_id == 1 else 'O'})")
        print(f"原始棋盘: {obs}")

        if player_id == -1:
            print(f"转换棋盘: {obs_transformed}")

        print(f"\n所有位置的概率预测:")
        print("  位置: ", end="")
        for i in range(9):
            print(f"{i:7}", end=" ")
        print()
        print("  概率: ", end="")
        for i in range(9):
            print(f"{probs[i]:7.4f}", end=" ")
        print()
        print("  状态: ", end="")
        for i in range(9):
            if i in legal_actions:
                print(f"{'合法':>7}", end=" ")
            else:
                print(f"{'非法':>7}", end=" ")
        print()

        print(f"\n合法动作: {legal_actions}")
        print(f"合法动作的概率:")
        for action in legal_actions:
            marker = " ← 最终选择" if action == final_action else ""
            print(f"  动作 {action}: Prob = {probs[action]:.4f}{marker}")

        print(f"\n最终选择: 动作 {final_action} (Prob = {probs[final_action]:.4f})")

        # 显示被过滤的高概率非法动作
        illegal_actions = [i for i in range(9) if i not in legal_actions]
        if illegal_actions:
            print(f"\n被过滤的非法动作:")
            for action in illegal_actions:
                print(f"  动作 {action}: Prob = {probs[action]:.4f} (已被占用)")

        print(f"{'='*60}\n")


def test_single_tree(model_path, n_games=50):
    """快速测试单棵树是否避免非法动作"""
    print("\n快速测试：单棵树 + 概率掩码是否避免非法动作\n")

    tree_player = SingleTreePlayer(model_path)
    minmax_player = MinMaxPlayer()

    results = battle_two_players(tree_player, minmax_player, n_games=n_games, verbose=False)
    print_battle_results(results, "单棵树", "MinMax")

    if results['player1_illegal'] == 0:
        print("\n✓ 成功：单棵树在所有测试中都避免了非法动作！")
        print("✓ 保持了完整的可解释性（单棵树）")
    else:
        print(f"\n✗ 单棵树仍有 {results['player1_illegal']} 次非法动作")


def main():
    parser = argparse.ArgumentParser(description="单棵树 + 概率掩码测试")

    parser.add_argument("--model-path", type=str,
                       default="log/viper_TicTacToe-v0_all-leaves_10_single_tree.joblib",
                       help="单棵树模型路径")

    parser.add_argument("--mode", type=str, default="test",
                       choices=['test', 'single', 'explain'],
                       help="测试模式")

    parser.add_argument("--n-games", type=int, default=50,
                       help="对战局数")

    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")

    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.mode == 'test':
        # 快速测试
        test_single_tree(args.model_path, n_games=args.n_games)

    elif args.mode == 'single':
        # 单局详细测试
        print("\n单局详细测试\n")
        env = TicTacToeBattleEnv()
        tree_player = SingleTreePlayer(args.model_path, debug=True, explain=True)
        minmax = MinMaxPlayer()

        obs = env.reset()
        step = 0

        while not env.done and step < 9:
            step += 1

            if env.current_player == 1:
                current_agent = tree_player
                agent_name = "单棵树 (X)"
                player_id = 1
            else:
                current_agent = minmax
                agent_name = "MinMax (O)"
                player_id = -1

            print(f"\n步骤 {step}: {agent_name} 的回合")
            print("当前棋盘:")
            env.render()

            if isinstance(current_agent, SingleTreePlayer):
                action = current_agent.predict(obs, player_id=player_id)
            else:
                action = current_agent.predict(obs)

            print(f"\n{agent_name} 选择动作: {action}")

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


if __name__ == "__main__":
    main()
