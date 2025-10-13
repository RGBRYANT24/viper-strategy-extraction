"""
修复对战脚本中的视角问题

问题：决策树训练时总是从X的视角（自己=1，对手=-1）
但对战时作为O，需要转换视角
"""

import sys
import re

def fix_battle_script():
    """修复battle_nn_vs_tree.py中的视角问题"""

    file_path = "battle_nn_vs_tree.py"

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 找到DecisionTreePlayer的predict方法
    old_predict = '''    def predict(self, obs):
        """预测动作"""
        obs_reshaped = obs.reshape(1, -1)
        action = self.model.predict(obs_reshaped)[0]

        # 调试输出前几次预测
        self.predict_count += 1
        if self.debug and self.predict_count <= 5:
            print(f"\\n[TREE DEBUG {self.predict_count}]")
            print(f"  输入棋盘: {obs}")
            print(f"  预测动作: {action}, 类型: {type(action)}")
            print(f"  合法动作: {np.where(obs == 0)[0]}")
            print(f"  动作是否合法: {action in np.where(obs == 0)[0]}")

        return action'''

    new_predict = '''    def predict(self, obs, player_id=1):
        """
        预测动作

        Args:
            obs: 棋盘状态（从当前环境视角）
            player_id: 当前玩家ID（1=X, -1=O），用于转换视角
        """
        # 重要：决策树训练时总是从X的视角（自己=1，对手=-1）
        # 如果现在是O玩家，需要翻转棋盘视角
        if player_id == -1:
            # 翻转视角：X变成对手(-1)，O变成自己(1)
            obs_transformed = -obs
        else:
            obs_transformed = obs

        obs_reshaped = obs_transformed.reshape(1, -1)
        action = self.model.predict(obs_reshaped)[0]

        # 调试输出前几次预测
        self.predict_count += 1
        if self.debug and self.predict_count <= 5:
            print(f"\\n[TREE DEBUG {self.predict_count}]")
            print(f"  玩家ID: {player_id}")
            print(f"  原始棋盘: {obs}")
            if player_id == -1:
                print(f"  转换后棋盘: {obs_transformed}")
            print(f"  预测动作: {action}, 类型: {type(action)}")
            print(f"  合法动作: {np.where(obs == 0)[0]}")
            print(f"  动作是否合法: {action in np.where(obs == 0)[0]}")

        return action'''

    if old_predict in content:
        content = content.replace(old_predict, new_predict)
        print("✓ 已修复 DecisionTreePlayer.predict() 方法")
    else:
        print("⚠ 未找到目标代码，可能已经被修改")
        return False

    # 修复battle_two_players函数中调用predict的地方
    old_battle_call = '''            # 根据当前玩家选择对应的智能体
            if env.current_player == 1:
                current_agent = player1
                agent_name = "Player1 (X)"
            else:
                current_agent = player2
                agent_name = "Player2 (O)"

            # 预测动作
            action = current_agent.predict(obs)'''

    new_battle_call = '''            # 根据当前玩家选择对应的智能体
            if env.current_player == 1:
                current_agent = player1
                agent_name = "Player1 (X)"
                current_player_id = 1
            else:
                current_agent = player2
                agent_name = "Player2 (O)"
                current_player_id = -1

            # 预测动作（如果是DecisionTreePlayer，传入player_id用于视角转换）
            if isinstance(current_agent, DecisionTreePlayer):
                action = current_agent.predict(obs, player_id=current_player_id)
            else:
                action = current_agent.predict(obs)'''

    if old_battle_call in content:
        content = content.replace(old_battle_call, new_battle_call)
        print("✓ 已修复 battle_two_players() 函数中的调用")
    else:
        print("⚠ 未找到battle调用代码，尝试部分匹配...")

    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\n✓ 文件已更新: {file_path}")
    print("\n请重新运行对战测试！")
    return True


if __name__ == "__main__":
    print("="*70)
    print("修复 battle_nn_vs_tree.py 中的视角问题")
    print("="*70)
    print()

    success = fix_battle_script()

    if success:
        print("\n" + "="*70)
        print("修复完成！")
        print("="*70)
        print("\n下一步：重新运行对战测试")
        print("python battle_nn_vs_tree.py --oracle-path log/oracle_TicTacToe_selfplay.zip --viper-path log/viper_TicTacToe-v0_100_15.joblib --mode both --n-games 200")
    else:
        print("\n❌ 修复失败，请手动修改")
        sys.exit(1)
